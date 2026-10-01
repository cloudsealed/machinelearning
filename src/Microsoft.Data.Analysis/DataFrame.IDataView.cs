// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Microsoft.Data.Analysis
{
    public partial class DataFrame : IDataView
    {
        bool IDataView.CanShuffle => true;

        private DataViewSchema _schema;
        private DataViewSchema DataViewSchema
        {
            get
            {
                if (_schema != null)
                {
                    return _schema;
                }

                var schemaBuilder = new DataViewSchema.Builder();
                for (int i = 0; i < Columns.Count; i++)
                {
                    DataFrameColumn baseColumn = Columns[i];
                    baseColumn.AddDataViewColumn(schemaBuilder);
                }
                _schema = schemaBuilder.ToSchema();
                return _schema;
            }
        }

        DataViewSchema IDataView.Schema => DataViewSchema;

        long? IDataView.GetRowCount() => Rows.Count;

        private DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            var activeColumns = new bool[DataViewSchema.Count];
            foreach (DataViewSchema.Column column in columnsNeeded)
            {
                if (column.Index < activeColumns.Length)
                {
                    activeColumns[column.Index] = true;
                }
            }

            return new RowCursor(this, activeColumns, 0, Rows.Count, 0, rand);
        }

        DataViewRowCursor IDataView.GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand)
        {
            return GetRowCursorCore(columnsNeeded, rand);
        }

        DataViewRowCursor[] IDataView.GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand)
        {
            var activeColumns = new bool[DataViewSchema.Count];
            foreach (DataViewSchema.Column column in columnsNeeded)
            {
                if (column.Index < activeColumns.Length)
                {
                    activeColumns[column.Index] = true;
                }
            }

            long rowCount = Rows.Count;
            n = Math.Max(1, Math.Min(n, (int)Math.Min(rowCount == 0 ? 1 : rowCount, int.MaxValue)));

            var cursors = new DataViewRowCursor[n];
            long baseSize = rowCount / n;
            long remainder = rowCount % n;
            long start = 0;
            for (int i = 0; i < n; i++)
            {
                long end = start + baseSize + (i < remainder ? 1 : 0);
                // Each partition gets a derived seed so Random state is not shared across threads.
                Random partitionRand = rand != null ? new Random(rand.Next()) : null;
                cursors[i] = new RowCursor(this, activeColumns, start, end, i, partitionRand);
                start = end;
            }
            return cursors;
        }

        private sealed class RowCursor : DataViewRowCursor
        {
            private bool _disposed;
            private long _position;
            private readonly DataFrame _dataFrame;
            private readonly Delegate[] _getters;
            private readonly long _startRow;
            private readonly long _endRow;
            private readonly long _batch;
            private readonly long[] _rowOrder; // non-null when shuffle is active

            public RowCursor(DataFrame dataFrame, bool[] activeColumns,
                             long startRow = 0, long endRow = -1, long batch = 0, Random rand = null)
            {
                Debug.Assert(dataFrame != null);
                Debug.Assert(activeColumns != null);

                _position = -1;
                _dataFrame = dataFrame;
                _startRow = startRow;
                _endRow = endRow < 0 ? dataFrame.Rows.Count : endRow;
                _batch = batch;

                if (rand != null && _endRow > _startRow)
                {
                    long count = _endRow - _startRow;
                    _rowOrder = new long[count];
                    for (long i = 0; i < count; i++) _rowOrder[i] = _startRow + i;
                    // Fisher-Yates shuffle
                    for (long i = count - 1; i > 0; i--)
                    {
                        long j = (long)rand.Next(0, (int)(i + 1));
                        (_rowOrder[i], _rowOrder[j]) = (_rowOrder[j], _rowOrder[i]);
                    }
                }

                _getters = new Delegate[Schema.Count];
                for (int i = 0; i < _getters.Length; i++)
                {
                    if (!activeColumns[i])
                        continue;
                    _getters[i] = CreateGetterDelegate(i);
                    Debug.Assert(_getters[i] != null);
                }
            }

            // Position must return the actual DataFrame row index because
            // GetDataViewGetter captures this cursor and uses Position as the column index.
            public override long Position => _rowOrder != null
                ? (_position >= 0 ? _rowOrder[_position] : -1L)
                : (_position >= 0 ? _startRow + _position : -1L);
            public override long Batch => _batch;
            public override DataViewSchema Schema => _dataFrame.DataViewSchema;

            protected override void Dispose(bool disposing)
            {
                if (_disposed)
                    return;
                if (disposing)
                {
                    _position = -1;
                }
                _disposed = true;
                base.Dispose(disposing);
            }

            private Delegate CreateGetterDelegate(int col)
            {
                DataFrameColumn column = _dataFrame.Columns[col];
                return column.GetDataViewGetter(this);
            }

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                if (!IsColumnActive(column))
                    throw new ArgumentOutOfRangeException(nameof(column));

                return (ValueGetter<TValue>)_getters[column.Index];
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return (ref DataViewRowId value) => value = new DataViewRowId((ulong)Position, 0);
            }

            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                return _getters[column.Index] != null;
            }

            public override bool MoveNext()
            {
                if (_disposed)
                    return false;
                _position++;
                long limit = _rowOrder?.Length ?? (_endRow - _startRow);
                return _position < limit;
            }
        }
    }
}
