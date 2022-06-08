use std::fmt::{self, Debug, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use std::str::FromStr;

/// Data trait used throughout the package
/// to control for floating point numbers.
pub trait MatrixData<T>:
    Mul<Output = T>
    + Display
    + Add<Output = T>
    + Div<Output = T>
    + Neg<Output = T>
    + Copy
    + Debug
    + PartialEq
    + PartialOrd
    + AddAssign
    + Sub<Output = T>
    + SubAssign
    + Sum
    + std::marker::Send
    + std::marker::Sync
{
    const ZERO: T;
    const ONE: T;
    const MIN: T;
    const MAX: T;
    const NAN: T;
    fn from_usize(v: usize) -> T;
    fn from_u16(v: u16) -> T;
    fn is_nan(self) -> bool;
    fn ln(self) -> T;
    fn exp(self) -> T;
}
impl MatrixData<f64> for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const MIN: f64 = f64::MIN;
    const MAX: f64 = f64::MAX;
    const NAN: f64 = f64::NAN;
    fn from_usize(v: usize) -> f64 {
        v as f64
    }
    fn from_u16(v: u16) -> f64 {
        f64::from(v)
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn ln(self) -> f64 {
        self.ln()
    }
    fn exp(self) -> f64 {
        self.exp()
    }
}

impl MatrixData<f32> for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    const MIN: f32 = f32::MIN;
    const MAX: f32 = f32::MAX;
    const NAN: f32 = f32::NAN;
    fn from_usize(v: usize) -> f32 {
        v as f32
    }
    fn from_u16(v: u16) -> f32 {
        f32::from(v)
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn ln(self) -> f32 {
        self.ln()
    }
    fn exp(self) -> f32 {
        self.exp()
    }
}

/// Contigious Column major matrix data container. This is
/// used throughout the crate, to house both the user provided data
/// as well as the binned data.
pub struct Matrix<'a, T> {
    pub data: &'a [T],
    pub index: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
    stride1: usize,
    stride2: usize,
}

impl<'a, T> Matrix<'a, T> {
    pub fn new(data: &'a [T], rows: usize, cols: usize) -> Self {
        Matrix {
            data,
            index: (0..rows).collect(),
            rows,
            cols,
            stride1: rows,
            stride2: 1,
        }
    }

    /// Get a single reference to an item in the matrix.
    ///
    /// * `i` - The ith row of the data to get.
    /// * `j` - the jth column of the data to get.
    pub fn get(&self, i: usize, j: usize) -> &T {
        &self.data[self.item_index(i, j)]
    }

    fn item_index(&self, i: usize, j: usize) -> usize {
        let mut idx: usize;
        idx = self.stride2 * i;
        idx += j * self.stride1;
        idx
    }

    /// Get a slice of a column in the matrix.
    ///
    /// * `col` - The index of the column to select.
    /// * `start_row` - The index of the start of the slice.
    /// * `end_row` - The index of the end of the slice of the column to select.
    pub fn get_col_slice(&self, col: usize, start_row: usize, end_row: usize) -> &[T] {
        let i = self.item_index(start_row, col);
        let j = self.item_index(end_row, col);
        &self.data[i..j]
    }

    /// Get an entire column in the matrix.
    ///
    /// * `col` - The index of the column to get.
    pub fn get_col(&self, col: usize) -> &[T] {
        self.get_col_slice(col, 0, self.rows)
    }
}

impl<'a, T> fmt::Display for Matrix<'a, T>
where
    T: FromStr + std::fmt::Display,
    <T as FromStr>::Err: 'static + std::error::Error,
{
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut val = String::new();
        for i in 0..self.rows {
            for j in 0..self.cols {
                val.push_str(self.get(i, j).to_string().as_str());
                if j == (self.cols - 1) {
                    val.push('\n');
                } else {
                    val.push(' ');
                }
            }
        }
        write!(f, "{}", val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_get() {
        let v = vec![1, 2, 3, 5, 6, 7];
        let m = Matrix::new(&v, 2, 3);
        println!("{}", m);
        assert_eq!(m.get(0, 0), &1);
        assert_eq!(m.get(1, 0), &2);
    }
    #[test]
    fn test_get_col_slice() {
        let v = vec![1, 2, 3, 5, 6, 7];
        let m = Matrix::new(&v, 3, 2);
        assert_eq!(m.get_col_slice(0, 0, 3), &vec![1, 2, 3]);
        assert_eq!(m.get_col_slice(1, 0, 2), &vec![5, 6]);
        assert_eq!(m.get_col_slice(1, 1, 3), &vec![6, 7]);
        assert_eq!(m.get_col_slice(0, 1, 2), &vec![2]);
    }

    #[test]
    fn test_get_col() {
        let v = vec![1, 2, 3, 5, 6, 7];
        let m = Matrix::new(&v, 3, 2);
        assert_eq!(m.get_col(1), &vec![5, 6, 7]);
    }
}
