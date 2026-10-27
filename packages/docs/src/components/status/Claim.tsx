import MathLabel from '@site/src/components/MathLabel'
import React, { type ReactNode } from 'react'

/**
 * How far a statement has been checked:
 *
 * - `reproduced`: a command of the Python package writes the record that the statement
 *   is computed from, and the record is in the repository;
 * - `preliminary`: checked only with code that is not yet part of the package, so no
 *   number from that check appears on these pages;
 * - `open`: not tested.
 */
export type ClaimStatus = 'reproduced' | 'preliminary' | 'open'

export const STATUS_LABELS: Record<ClaimStatus, string> = {
  reproduced: 'reproduced by this package',
  preliminary: 'preliminary',
  open: 'open'
}

interface ClaimProps {
  readonly status: ClaimStatus
  /** The statement, in a few words, with inline math between dollar signs. */
  readonly title: string
  /** The command that writes the record behind the statement, if any. */
  readonly command?: string
  readonly children: ReactNode
}

/** One statement of the status board, with how far it has been checked and the evidence. */
export default function Claim({ status, title, command, children }: ClaimProps): ReactNode {
  return (
    <section className={`claim claim--${status}`}>
      <header className="claim__header">
        <span className="claim__label">{STATUS_LABELS[status]}</span>
        <span className="claim__title">
          <MathLabel label={title} />
        </span>
      </header>
      <div className="claim__body">{children}</div>
      {command === undefined ? null : (
        <p className="claim__command">
          <code>{command}</code>
        </p>
      )}
    </section>
  )
}
