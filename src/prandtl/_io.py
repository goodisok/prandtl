"""CFD data I/O utilities.

Read OpenFOAM forces output and SU2 history files into numpy arrays
with automatic column detection.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np  # noqa: I001

# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #


def _resolve_foam_path(path: str | Path) -> Path:
    """Resolve *path* to an actual forces data file.

    If *path* is a directory, search it for ``coefficient.dat`` or
    ``forces.dat``.
    """
    p = Path(os.path.expanduser(path))

    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    if p.is_file():
        return p

    if p.is_dir():
        for candidate in ("coefficient.dat", "forces.dat"):
            candidate_path = p / candidate
            if candidate_path.is_file():
                return candidate_path
        raise FileNotFoundError(
            f"No forces file found in directory {p}. Expected 'coefficient.dat' or 'forces.dat'."
        )

    raise FileNotFoundError(f"Path is neither a file nor a directory: {p}")


def _clean_column_name(raw: str) -> str:
    """Clean an OpenFOAM-style column name.

    Removes leading ``#``, strips surrounding parentheses, and discards
    empty tokens.
    """
    name = raw.strip()
    # Remove leading '#' if present
    if name.startswith("#"):
        name = name[1:].strip()
    # Strip outer parentheses: (Cd) -> Cd,  forces(pressure) -> forces(pressure)
    # We only strip a *single* pair of matching outer parens.
    if name.startswith("(") and name.endswith(")"):
        name = name[1:-1]
    return name


def _parse_foam_columns(header_line: str) -> list[str]:
    """Parse column names from an OpenFOAM header line.

    Handles tab-separated columns that may be wrapped in parentheses.

    Parameters
    ----------
    header_line : str
        A raw header line (without leading ``#``).

    Returns
    -------
    list of str
        Cleaned column names.
    """
    # Split on whitespace (tabs or spaces)
    raw_names = header_line.split()
    return [_clean_column_name(n) for n in raw_names if n]


def _load_foam_data(file_path: Path) -> tuple[list[str], np.ndarray]:
    """Load column names and numeric data from an OpenFOAM forces file.

    Returns
    -------
    columns : list of str
        Detected column names.
    data : ndarray
        Numeric data matrix.
    """
    lines: list[str] = []
    try:
        with open(file_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot read file {file_path}: {exc}") from exc

    # Separate comment lines from data lines
    comment_lines: list[str] = []
    data_lines: list[str] = []

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            comment_lines.append(stripped)
        else:
            data_lines.append(stripped)

    if not comment_lines:
        raise ValueError(
            f"No header lines (starting with '#') found in {file_path}. "
            f"Cannot determine column names."
        )

    # The last comment line before data is the column header
    raw_header = comment_lines[-1]
    # Strip the leading '#' for parsing
    if raw_header.startswith("#"):
        raw_header = raw_header[1:]

    columns = _parse_foam_columns(raw_header)

    if not columns:
        raise ValueError(f"Could not parse column names from header in {file_path}.")

    if not data_lines:
        raise ValueError(f"No data rows found in {file_path}.")

    # Parse numeric data
    rows: list[list[float]] = []
    for lineno, line in enumerate(data_lines, start=1):
        tokens = line.split()
        try:
            row = [float(t) for t in tokens]
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric value in {file_path} data line {lineno}: {line.strip()!r}"
            ) from exc
        rows.append(row)

    # Check column consistency
    if rows:
        n_cols = len(columns)
        for i, row in enumerate(rows):
            if len(row) != n_cols:
                raise ValueError(
                    f"Column count mismatch in {file_path} line {i + 1}: "
                    f"expected {n_cols} columns (based on header), "
                    f"got {len(row)}."
                )

    return columns, np.array(rows, dtype=np.float64)


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #


def read_foam_forces(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Parse an OpenFOAM forces output file.

    Reads ``coefficient.dat`` or ``forces.dat`` produced by the OpenFOAM
    ``forces`` function object.  Handles ``#`` comment lines and
    bracket-wrapped column names such as ``(Cd)`` or ``forces(pressure)``.

    If *path* is a directory, the function automatically searches for
    ``coefficient.dat`` or ``forces.dat`` inside it.

    Parameters
    ----------
    path : str or Path
        Path to the forces data file, or a directory containing one.

    Returns
    -------
    X : ndarray of shape (n_rows, 1)
        Input variable column (e.g. time or iteration step).
        The **first** data column is always treated as the independent
        variable.
    Y : ndarray of shape (n_rows, n_outputs)
        Output columns (e.g. Cd, Cl, Cm).  All columns beyond the first.
    params : list of str
        Column name for *X* (single element).
    outputs : list of str
        Column names for *Y*.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist, or no forces file is found inside a
        directory.
    ValueError
        If the file contains no header, no data rows, or has malformed
        numeric values.

    Examples
    --------
    >>> X, Y, params, outputs = read_foam_forces("postProcessing/forces/0/coefficient.dat")
    >>> X[:3]
    array([[0.1],
           [0.2],
           [0.3]])
    >>> params
    ['Time']
    >>> outputs
    ['Cd', 'Cl', 'Cm']
    """
    file_path = _resolve_foam_path(path)
    columns, data = _load_foam_data(file_path)

    # First column -> X (params), rest -> Y (outputs)
    X = data[:, :1]  # noqa: N806  shape (n, 1)
    Y = data[:, 1:]  # noqa: N806  shape (n, m)

    params = columns[:1]
    outputs = columns[1:]

    return X, Y, params, outputs


def read_su2_history(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Parse a SU2 history CSV file.

    Reads ``history.csv`` produced by SU2 solvers.  The file must contain a
    header row with quoted or unquoted column names, followed by numeric
    data rows.

    The **first** column is treated as the independent variable (typically
    ``"Iteration"`` or ``"Time_Iter"``).  All remaining columns are treated
    as outputs.

    Parameters
    ----------
    path : str or Path
        Path to the ``history.csv`` file.  Unlike :func:`read_foam_forces`,
        directories are not auto-resolved.

    Returns
    -------
    X : ndarray of shape (n_rows, 1)
        Independent variable column (e.g. iteration number).
    Y : ndarray of shape (n_rows, n_outputs)
        Output columns (e.g. CL, CD, CMz).
    params : list of str
        Column name for *X* (single element).
    outputs : list of str
        Column names for *Y*.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file contains no header, no data rows, or has malformed
        numeric values.

    Examples
    --------
    >>> X, Y, params, outputs = read_su2_history("history.csv")
    >>> params
    ['Iteration']
    >>> outputs
    ['CL', 'CD', 'CMz']
    """
    file_path = Path(os.path.expanduser(path))

    if not file_path.exists():
        raise FileNotFoundError(f"Path does not exist: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    try:
        with open(file_path, encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            rows_raw = list(reader)
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot read file {file_path}: {exc}") from exc

    if not rows_raw:
        raise ValueError(f"File is empty: {file_path}")

    # First row is header
    header = [h.strip().strip('"').strip("'") for h in rows_raw[0]]
    if not header or all(h == "" for h in header):
        raise ValueError(f"No column headers found in {file_path}.")

    data_rows_raw = rows_raw[1:]

    if not data_rows_raw:
        raise ValueError(f"No data rows found in {file_path}.")

    # Parse numeric data
    rows: list[list[float]] = []
    for lineno, row in enumerate(data_rows_raw, start=2):
        try:
            rows.append([float(v) for v in row])
        except ValueError as exc:
            raise ValueError(f"Non-numeric value in {file_path} line {lineno}: {row!r}") from exc

    n_cols = len(header)
    for i, row in enumerate(rows):
        if len(row) != n_cols:
            raise ValueError(
                f"Column count mismatch in {file_path} line {i + 2}: "
                f"expected {n_cols} columns (based on header), "
                f"got {len(row)}."
            )

    data = np.array(rows, dtype=np.float64)

    X = data[:, :1]  # noqa: N806
    Y = data[:, 1:]  # noqa: N806

    params = header[:1]
    outputs = header[1:]

    return X, Y, params, outputs
