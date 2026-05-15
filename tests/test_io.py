"""Tests for CFD data parsers.

Validates OpenFOAM forces output and SU2 history.csv parsing.
"""

import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

import prandtl as pr


# ------------------------------------------------------------------ #
#  read_foam_forces
# ------------------------------------------------------------------ #


class TestReadFoamForces:
    """Test OpenFOAM forces output parsing."""

    def test_standard_format(self) -> None:
        content = textwrap.dedent("""\
        # Time               	Cd               	Cl               	Cm
        0.1				0.0234			0.5612			-0.0342
        0.2				0.0238			0.5589			-0.0338
        0.3				0.0241			0.5571			-0.0331
        """)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dat", delete=False
        ) as f:
            f.write(content)
            path = f.name

        try:
            X, Y, params, outputs = pr.read_foam_forces(path)
            np.testing.assert_array_equal(params, ["Time"])
            np.testing.assert_array_equal(outputs, ["Cd", "Cl", "Cm"])
            assert X.shape == (3, 1)
            assert Y.shape == (3, 3)
            np.testing.assert_almost_equal(X[2, 0], 0.3)
            np.testing.assert_almost_equal(Y[0, 0], 0.0234)
            np.testing.assert_almost_equal(Y[0, 1], 0.5612)
            np.testing.assert_almost_equal(Y[0, 2], -0.0342)
        finally:
            Path(path).unlink()

    def test_paren_column_names(self) -> None:
        """OpenFOAM sometimes wraps column names in parentheses."""
        content = textwrap.dedent("""\
        # Time              	(Cd)             	(Cl)             	(Cm)
        0.1				0.0234			0.5612			-0.0342
        """)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dat", delete=False
        ) as f:
            f.write(content)
            path = f.name

        try:
            X, Y, params, outputs = pr.read_foam_forces(path)
            assert outputs == ["Cd", "Cl", "Cm"]
        finally:
            Path(path).unlink()

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            pr.read_foam_forces("/nonexistent/path/forces.dat")

    def test_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dat", delete=False
        ) as f:
            f.write("# just a header\n")

        try:
            with pytest.raises(ValueError, match="data rows"):
                pr.read_foam_forces(f.name)
        finally:
            Path(f.name).unlink()


# ------------------------------------------------------------------ #
#  read_su2_history
# ------------------------------------------------------------------ #


class TestReadSu2History:
    """Test SU2 history.csv parsing."""

    def test_standard_format(self) -> None:
        content = textwrap.dedent("""\
        "Time","Iter","CL","CD"
        0.1,1,0.5612,0.0234
        0.2,2,0.5589,0.0238
        0.3,3,0.5571,0.0241
        """)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(content)
            path = f.name

        try:
            X, Y, params, outputs = pr.read_su2_history(path)
            np.testing.assert_array_equal(params, ["Time"])
            assert outputs == ["Iter", "CL", "CD"]
            assert X.shape == (3, 1)
            assert Y.shape == (3, 3)
            np.testing.assert_almost_equal(X[0, 0], 0.1)
            np.testing.assert_almost_equal(Y[0, 1], 0.5612)
        finally:
            Path(path).unlink()

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            pr.read_su2_history("/nonexistent/path/history.csv")

    def test_not_enough_columns(self) -> None:
        content = "a,b\n1,2\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(content)
            path = f.name

        try:
            X, Y, params, outputs = pr.read_su2_history(path)
            assert Y.shape[1] == 1, "Should have just 1 output column"
        finally:
            Path(path).unlink()