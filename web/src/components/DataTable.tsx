import type { ReactNode } from 'react';
import type { TableRow } from '../lib/data';
import { cx } from '../lib/format';

export interface Column {
  key: string;
  label: ReactNode;
  align?: 'left' | 'right';
  render: (row: TableRow) => ReactNode;
}

/** Compact, horizontally scrollable table used by the Results sections. */
export function DataTable({ columns, rows, caption, highlight, minWidth = 480 }: { columns: Column[]; rows: TableRow[]; caption: string; highlight?: (row: TableRow) => boolean; minWidth?: number }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[13px]" style={{ minWidth }}>
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-y border-line bg-surface-2/60 text-left text-[11px] tracking-wide text-fg-3 uppercase">
            {columns.map((c, i) => (
              <th key={c.key} scope="col" className={cx('py-2 font-medium', i === 0 ? 'pr-3 pl-5' : 'px-3', i === columns.length - 1 && 'pr-5', c.align === 'left' ? 'text-left' : i === 0 ? 'text-left' : 'text-right')}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="tnum">
          {rows.map((r, ri) => (
            <tr key={ri} className={cx('border-b border-line last:border-0', highlight?.(r) && 'bg-accent-soft')}>
              {columns.map((c, i) =>
                i === 0 ? (
                  <th key={c.key} scope="row" className="py-2 pr-3 pl-5 text-left font-medium text-fg">
                    {c.render(r)}
                  </th>
                ) : (
                  <td key={c.key} className={cx('px-3 py-2 text-fg-2', i === columns.length - 1 && 'pr-5', c.align === 'left' ? 'text-left' : 'text-right')}>
                    {c.render(r)}
                  </td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
