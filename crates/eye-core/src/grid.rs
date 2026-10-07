use nalgebra::Point2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    cols: u32,
    rows: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cell {
    pub col: u32,
    pub row: u32,
}

impl Grid {
    pub fn new(cols: u32, rows: u32) -> Option<Self> {
        (cols > 0 && rows > 0).then_some(Self { cols, rows })
    }

    pub fn cols(&self) -> u32 {
        self.cols
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Cell holding `p` on an output of `size` (same units as `p`). Boundaries belong to the cell
    /// on their right/bottom. `None` outside `[0, w) x [0, h)` or for non-finite input.
    pub fn cell_of(&self, p: Point2<f64>, (w, h): (f64, f64)) -> Option<Cell> {
        if !(p.x.is_finite() && p.y.is_finite()) || p.x < 0.0 || p.y < 0.0 || p.x >= w || p.y >= h {
            return None;
        }
        let col = ((p.x * f64::from(self.cols) / w).floor() as u32).min(self.cols - 1);
        let row = ((p.y * f64::from(self.rows) / h).floor() as u32).min(self.rows - 1);
        Some(Cell { col, row })
    }

    pub fn cell_bounds(&self, cell: Cell, (w, h): (f64, f64)) -> (Point2<f64>, Point2<f64>) {
        let (cw, ch) = (w / f64::from(self.cols), h / f64::from(self.rows));
        (
            Point2::new(f64::from(cell.col) * cw, f64::from(cell.row) * ch),
            Point2::new(f64::from(cell.col + 1) * cw, f64::from(cell.row + 1) * ch),
        )
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    const OUTPUT: (f64, f64) = (1920.0, 1080.0);

    #[test]
    fn test_grid_new_rejects_zero() {
        assert!(Grid::new(0, 3).is_none());
        assert!(Grid::new(3, 0).is_none());
    }

    #[test]
    fn test_cell_of_corners_4x4_on_1920x1080() {
        let grid = Grid::new(4, 4).unwrap();
        assert_eq!(
            grid.cell_of(Point2::new(0.0, 0.0), OUTPUT),
            Some(Cell { col: 0, row: 0 })
        );
        assert_eq!(
            grid.cell_of(Point2::new(1919.999, 1079.999), OUTPUT),
            Some(Cell { col: 3, row: 3 })
        );
    }

    #[test]
    fn test_cell_of_boundary_belongs_to_right_bottom_cell() {
        let grid = Grid::new(4, 4).unwrap();
        assert_eq!(
            grid.cell_of(Point2::new(480.0, 270.0), OUTPUT),
            Some(Cell { col: 1, row: 1 })
        );
        assert_eq!(
            grid.cell_of(Point2::new(479.999, 269.999), OUTPUT),
            Some(Cell { col: 0, row: 0 })
        );
    }

    #[test]
    fn test_cell_of_outside_or_nonfinite_is_none() {
        let grid = Grid::new(4, 4).unwrap();
        assert_eq!(grid.cell_of(Point2::new(1920.0, 0.0), OUTPUT), None);
        assert_eq!(grid.cell_of(Point2::new(-0.001, 5.0), OUTPUT), None);
        assert_eq!(grid.cell_of(Point2::new(5.0, 1080.0), OUTPUT), None);
        assert_eq!(grid.cell_of(Point2::new(f64::NAN, 5.0), OUTPUT), None);
    }

    proptest! {
        #[test]
        fn test_cell_bounds_contain_point(cols in 1u32..=8, rows in 1u32..=8, x in 0.0f64..1920.0, y in 0.0f64..1080.0) {
            let grid = Grid::new(cols, rows).unwrap();
            let p = Point2::new(x, y);
            let cell = grid.cell_of(p, OUTPUT).unwrap();
            let (min, max) = grid.cell_bounds(cell, OUTPUT);
            prop_assert!(min.x <= p.x && p.x < max.x);
            prop_assert!(min.y <= p.y && p.y < max.y);
        }
    }
}
