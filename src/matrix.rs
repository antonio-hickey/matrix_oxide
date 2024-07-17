use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

/// MxN Matrix
pub struct Matrix<T> {
    pub data: Vec<T>,
    pub row_size: usize,
    pub col_size: usize,
}
impl<T: Default + Clone> Matrix<T> {
    /// Construct a new *non-empty* and *sized* `Matrix`
    pub fn new(row_size: usize, col_size: usize) -> Self {
        Matrix {
            data: vec![T::default(); row_size * col_size],
            row_size,
            col_size,
        }
    }

    /// Try to get a reference to the value at a given row and column from the matrix
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.row_size && col < self.col_size {
            Some(&self.data[col + row * self.col_size])
        } else {
            None
        }
    }

    /// Get a vector of the diagonal elements of the matrix
    pub fn get_diagonal(&self) -> Vec<T> {
        (0..self.col_size)
            .filter_map(|col_idx| self.get(col_idx, col_idx).cloned())
            .collect()
    }

    /// Try to get a mutable reference to the value at a given row and column from the matrix
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.row_size && col < self.col_size {
            Some(&mut self.data[col + row * self.col_size])
        } else {
            None
        }
    }

    /// Try to set a value at a given row and column in the matrix
    pub fn set(&mut self, row: usize, column: usize, value: T) -> bool {
        if let Some(cell) = self.get_mut(row, column) {
            *cell = value;
            true
        } else {
            false
        }
    }

    /// Try to get all the values for a given column
    ///
    /// NOTE: If you pass a column value larger than the number of columns
    /// this function will return None.
    pub fn try_get_column(&self, column: usize) -> Option<Vec<T>> {
        // Bounds check
        if column >= self.col_size {
            return None;
        }

        // Iterate over all the rows grabbing a specific column each time
        let col_data: Vec<T> = (0..self.row_size)
            .map(|row| self.data[row * self.col_size + column].clone())
            .collect();

        Some(col_data)
    }

    /// Try to get all the values for a given row
    ///
    /// NOTE: If you pass a row value larger than the number of rows
    /// this function will return None.
    pub fn try_get_row(&self, row: usize) -> Option<Vec<T>> {
        // Bounds check
        if row >= self.row_size {
            return None;
        }

        // Iterate over all the rows grabbing a specific column each time
        let row_data: Vec<T> = (0..self.col_size)
            .map(|col| self.data[row * self.col_size + col].clone())
            .collect();

        Some(row_data)
    }

    /// Create a `Matrix` from a columns (vec of vec)
    pub fn from_columns(cols: Vec<Vec<T>>) -> Matrix<T> {
        if cols.is_empty() {
            return Matrix {
                data: Vec::new(),
                row_size: 0,
                col_size: 0,
            };
        }

        let row_size = cols[0].len();
        let col_size = cols.len();

        let data = (0..row_size)
            .flat_map(|row| cols.iter().filter_map(move |col| col.get(row).cloned()))
            .collect();

        Matrix {
            data,
            row_size,
            col_size,
        }
    }

    /// Create a sub matrix with a specific row and column to exclude
    pub fn sub_matrix(&self, skip_row: usize, skip_col: usize) -> Matrix<T> {
        let columns: Vec<Vec<T>> = (0..self.col_size)
            .filter_map(|col| {
                if col != skip_col {
                    Some(
                        self.try_get_column(col)
                            // TODO: handle this unwrap, im too tired rn
                            .unwrap()
                            .into_iter()
                            .enumerate()
                            .filter_map(|(row, val)| if row != skip_row { Some(val) } else { None })
                            .collect(),
                    )
                } else {
                    None
                }
            })
            .collect();

        Matrix::from_columns(columns)
    }

    /// Perform a transpose operation (swap rows for columns and vice versa)
    pub fn transpose(&self) -> Matrix<T> {
        Matrix {
            data: (0..self.col_size)
                .flat_map(|col| {
                    (0..self.row_size).map(move |row| self.data[row * self.col_size + col].clone())
                })
                .collect(),

            row_size: self.col_size,
            col_size: self.row_size,
        }
    }
}

impl<T: Default + Clone> Default for Matrix<T> {
    /// Create a default `Matrix` instance
    fn default() -> Self {
        Self::new(2, 3)
    }
}
impl<T: Default + Clone + Debug> Add for Matrix<T>
where
    T: Add<Output = T> + Clone,
{
    type Output = Matrix<T>;

    /// Matrix addition
    /// NOTE: the matrices you add MUST have the same dimensionality
    fn add(self, rhs: Self) -> Matrix<T> {
        let data: Vec<T> = (0..self.row_size)
            .flat_map(|row| {
                let row_a = self.try_get_row(row).expect("Invalid row in self");
                let row_b = rhs.try_get_row(row).expect("Invalid row in rhs");
                row_a.into_iter().zip(row_b).map(|(a, b)| a + b)
            })
            .collect();

        Matrix {
            data,
            col_size: self.col_size,
            row_size: self.row_size,
        }
    }
}
impl<T> Matrix<T>
where
    T: Default + Clone + Add<Output = T>,
{
    /// Perform the trace operation that computes the sum of all diagonal
    /// elements in the matrix.
    ///
    /// NOTE: off-diagnonal elements do NOT contribute to the trace of the
    /// matrix, so 2 very different matrices can have the same trace.
    pub fn trace(&self) -> T {
        self.get_diagonal()
            .into_iter()
            .fold(T::default(), |acc, diagonal| acc + diagonal)
    }
}
impl<T: Default + Clone + Debug> Sub for Matrix<T>
where
    T: Sub<Output = T> + Clone,
{
    type Output = Matrix<T>;

    /// Subtract a matrix by another matrix
    /// NOTE: the matrix you subtract by MUST have the same dimensionality
    fn sub(self, rhs: Self) -> Matrix<T> {
        let data: Vec<T> = (0..self.row_size)
            .flat_map(|row| {
                let row_a = self.try_get_row(row).expect("Invalid row in self");
                let row_b = rhs.try_get_row(row).expect("Invalid row in rhs");
                row_a.into_iter().zip(row_b).map(|(a, b)| a - b)
            })
            .collect();

        Matrix {
            data,
            row_size: self.row_size,
            col_size: self.col_size,
        }
    }
}
impl<T: Default> Matrix<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Clone,
{
    /// Multiply a matrix by a single number (scalar)
    /// NOTE: The scalar type MUST match the matrix type.
    pub fn scalar_multiply(&self, scalar: T) -> Matrix<T> {
        let data = self
            .data
            .iter()
            .map(|value| value.clone() * scalar.clone())
            .collect();

        Matrix {
            data,
            row_size: self.row_size,
            col_size: self.col_size,
        }
    }

    /// Multiply `Matrix` with another `Matrix` using standard matrix multiplication
    /// NOTE: The matrices inner dimensions MUST match else returns None
    pub fn multiply(&self, multiplier: &Matrix<T>) -> Option<Matrix<T>> {
        // Validity check for the matrices inner dimensions
        if self.col_size != multiplier.row_size {
            return None;
        }

        let data: Vec<T> = (0..self.row_size)
            .flat_map(|i| {
                (0..multiplier.col_size).map(move |j| {
                    (0..self.col_size)
                        .map(|k| {
                            self.data[i * self.col_size + k].clone()
                                * multiplier.data[k * multiplier.col_size + j].clone()
                        })
                        .fold(T::default(), |acc, x| acc + x)
                })
            })
            .collect();

        Some(Matrix {
            data,
            col_size: multiplier.col_size,
            row_size: self.row_size,
        })
    }

    /// Multiply the `Matrix` by a vector
    /// NOTE: The vectors length MUST match the vector columns, else returns None
    pub fn vector_multiply(&self, multiplier: &[T]) -> Option<Vec<T>> {
        // Validity check that the `Matrix` column size matches the vector column size
        if self.col_size != multiplier.len() {
            return None;
        }

        let data: Vec<T> = (0..self.row_size)
            .map(|i| {
                (0..multiplier.len())
                    .map(|j| self.data[i * self.col_size + j].clone() * multiplier[j].clone())
                    .fold(T::default(), |acc, x| acc + x)
            })
            .collect();

        Some(data)
    }

    /// Compute a unique determinant for a `Matrix`
    /// NOTE: Only computable for square (M x M) matrices.
    /// NOTE: The determinant is 0 for a `Matrix` with rank r < M (non-invertable).
    pub fn determinant(&self) -> Option<T> {
        // Validity check that it's a square matrix
        if self.col_size != self.row_size {
            return None;
        }

        // Base case for recursion: 1x1 matrix
        if self.row_size == 1 {
            return Some(self.data[0].clone());
        }

        if let Some(first_row) = self.try_get_row(0) {
            let determinant =
                first_row
                    .iter()
                    .enumerate()
                    .fold(T::default(), |acc, (col_idx, item)| {
                        let sub_matrix = self.sub_matrix(0, col_idx);
                        let sub_determinant = sub_matrix.determinant().unwrap();

                        // Alternate between addition and subtraction
                        // operations every iteration
                        if col_idx % 2 == 0 {
                            acc + item.clone() * sub_determinant
                        } else {
                            acc - item.clone() * sub_determinant
                        }
                    });

            Some(determinant)
        } else {
            None
        }
    }
}
impl<T> Matrix<T>
where
    T: Default + Clone + From<f64>,
{
    /// Create an identity matrix of given size
    pub fn identity(size: usize) -> Matrix<T> {
        let data = (0..size * size)
            .map(|i| {
                if i % (size + 1) == 0 {
                    T::from(1.0)
                } else {
                    T::default()
                }
            })
            .collect();

        Matrix {
            data,
            row_size: size,
            col_size: size,
        }
    }
}
impl<T> Matrix<T>
where
    T: Default + Clone + Mul<Output = T> + Add<Output = T> + Into<f64>,
{
    /// Compute the frobenius norm of a `Matrix`
    pub fn frobenius_norm(&self) -> f64 {
        let sum_of_squares: f64 = self
            .data
            .iter()
            .map(|val| {
                let val_f64: f64 = val.clone().into();
                val_f64 * val_f64
            })
            .fold(f64::default(), |acc, x| acc + x);

        sum_of_squares.sqrt()
    }
}
impl<T> Matrix<T>
where
    T: Copy
        + PartialOrd
        + Default
        + From<f64>
        + Into<f64>
        + Sub<Output = T>
        + Add<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Abs,
{
    /// Compute the inverse of a `Matrix`
    /// NOTE: Only computable for square (M x M) matrices.
    /// NOTE: Only computable for a `Matrix` where r = M (full rank).
    pub fn inverse(&self) -> Option<Matrix<T>> {
        // Validity check that it's a square matrix
        if self.row_size != self.col_size {
            return None;
        }

        let n = self.row_size;
        let epsilon = T::from(1e-10);

        // Regular matrix
        let mut a = self.data.clone();
        // Inverted matrix
        let mut b = Matrix::<T>::identity(n).data;

        for i in 0..n {
            let pivot = (i..n).fold(i, |acc, j| {
                match a[j * n + i].abs() > a[acc * n + i].abs() {
                    true => j,
                    false => acc,
                }
            });

            // Validity check that it's not a singular matrix
            if a[pivot * n + i].abs() < epsilon {
                return None;
            }

            if pivot != i {
                (0..n).for_each(|k| {
                    a.swap(i * n + k, pivot * n + k);
                    b.swap(i * n + k, pivot * n + k);
                });
            }

            let pivot_val = a[i * n + i];
            (0..n).for_each(|k| {
                a[i * n + k] = a[i * n + k] / pivot_val;
                b[i * n + k] = b[i * n + k] / pivot_val;
            });

            (0..n).filter(|&j| j != i).for_each(|j| {
                let factor = a[j * n + i];
                (0..n).for_each(|k| {
                    a[j * n + k] = a[j * n + k] - factor * a[i * n + k];
                    b[j * n + k] = b[j * n + k] - factor * b[i * n + k];
                })
            })
        }

        Some(Matrix {
            data: b,
            row_size: n,
            col_size: n,
        })
    }
}

pub trait Abs {
    fn abs(self) -> Self;
}
impl Abs for f64 {
    fn abs(self) -> Self {
        self.abs()
    }
}
impl Abs for i32 {
    fn abs(self) -> Self {
        self.abs()
    }
}
impl Abs for i64 {
    fn abs(self) -> Self {
        self.abs()
    }
}
impl Abs for f32 {
    fn abs(self) -> Self {
        self.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Check if 2 float value's are *ABOUT* equal
    fn approx_equal(a: &[f64], b: &[f64], epsilon: f64) -> bool {
        a.iter()
            .zip(b.iter())
            .all(|(&a, &b)| (a - b).abs() < epsilon)
    }

    #[test]
    fn test_matrix_vector_multiplication() {
        let matrix = Matrix::<i32> {
            data: vec![1, 2, 3, 4],
            row_size: 2,
            col_size: 2,
        };
        let vector = vec![5, 6];

        let expected = vec![17, 39];
        let result = matrix.vector_multiply(&vector).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_multiplication() {
        let matrix_a = Matrix::<i32> {
            data: vec![1, 2, 3, 4],
            row_size: 2,
            col_size: 2,
        };

        let matrix_b = Matrix::<i32> {
            data: vec![2, 0, 1, 2],
            row_size: 2,
            col_size: 2,
        };

        let expected = Matrix::<i32> {
            data: vec![4, 4, 10, 8],
            row_size: 2,
            col_size: 2,
        };
        let result = matrix_a.multiply(&matrix_b).unwrap();
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_martrix_trace() {
        let matrix = Matrix::<i32> {
            data: vec![1, 2, 3, 4],
            col_size: 2,
            row_size: 2,
        };

        let expected: i32 = 5;
        let result = matrix.trace();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_martrix_diagonal() {
        let matrix = Matrix::<i32> {
            data: vec![1, 2, 3, 4],
            row_size: 2,
            col_size: 2,
        };

        let expected: Vec<i32> = vec![1, 4];
        let result = matrix.get_diagonal();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_scalar_multiplication() {
        let matrix = Matrix::<i32> {
            data: vec![1, 2, 3, 4],
            row_size: 2,
            col_size: 2,
        };

        let expected = Matrix::<i32> {
            data: vec![2, 4, 6, 8],
            row_size: 2,
            col_size: 2,
        };
        let result = matrix.scalar_multiply(2);

        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_matrix_subtraction() {
        let matrix_a = Matrix::<i32> {
            data: vec![1, 2, 3, 4, 5, 6],
            row_size: 2,
            col_size: 3,
        };
        let matrix_b = Matrix::<i32> {
            data: vec![6, 5, 4, 3, 2, 1],
            row_size: 2,
            col_size: 3,
        };

        let expected = Matrix::<i32> {
            data: vec![-5, -3, -1, 1, 3, 5],
            row_size: 2,
            col_size: 3,
        };
        let result = matrix_a - matrix_b;
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_matrix_addition() {
        let matrix_a = Matrix::<i32> {
            data: vec![1, 2, 3, 4],
            row_size: 2,
            col_size: 2,
        };
        let matrix_b = Matrix::<i32> {
            data: vec![4, 3, 2, 1],
            row_size: 2,
            col_size: 2,
        };

        let expected = Matrix::<i32> {
            data: vec![5, 5, 5, 5],
            row_size: 2,
            col_size: 2,
        };

        let result = matrix_a + matrix_b;
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn new_matrix_has_correct_size() {
        let matrix: Matrix<i32> = Matrix::new(2, 3);
        assert_eq!(matrix.data.len(), 6);
    }

    #[test]
    fn default_matrix_is_same_as_new() {
        let default_matrix: Matrix<i32> = Matrix::default();
        let new_matrix: Matrix<i32> = Matrix::new(2, 3);
        assert_eq!(default_matrix.data, new_matrix.data);
    }

    #[test]
    fn try_get_column_valid() {
        let matrix: Matrix<i32> = Matrix::new(2, 3);
        let column = matrix.try_get_column(1);
        assert!(column.is_some());
        assert_eq!(column.unwrap(), vec![0, 0]);
    }

    #[test]
    fn try_get_column_invalid() {
        let matrix: Matrix<i32> = Matrix::new(2, 3);
        let column = matrix.try_get_column(3);
        assert!(column.is_none());
    }

    #[test]
    fn try_get_row_valid() {
        let matrix: Matrix<i32> = Matrix::new(2, 3);
        let row = matrix.try_get_row(0);
        assert!(row.is_some());
        assert_eq!(row.unwrap(), vec![0, 0, 0]);
    }

    #[test]
    fn try_get_row_invalid() {
        let matrix: Matrix<i32> = Matrix::new(2, 3);
        let row = matrix.try_get_row(2);
        assert!(row.is_none());
    }

    #[test]
    fn transpose_works_correctly() {
        let mut matrix: Matrix<i32> = Matrix::new(2, 3);
        for i in 0..matrix.data.len() {
            matrix.data[i] = i as i32;
        }
        let transposed = matrix.transpose();
        assert_eq!(transposed.data, vec![0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn determinant_of_1x1_matrix() {
        let matrix = Matrix {
            data: vec![7],
            row_size: 1,
            col_size: 1,
        };
        assert_eq!(matrix.determinant(), Some(7));
    }

    #[test]
    fn determinant_of_2x2_matrix() {
        let matrix = Matrix {
            data: vec![1, 2, 3, 4],
            row_size: 2,
            col_size: 2,
        };
        assert_eq!(matrix.determinant(), Some(-2));
    }

    #[test]
    fn determinant_of_3x3_matrix() {
        let matrix = Matrix {
            data: vec![3, 2, 1, 0, 1, 4, 5, 6, 0],
            row_size: 3,
            col_size: 3,
        };
        let expected = -37;
        assert_eq!(matrix.determinant(), Some(expected));
    }

    #[test]
    fn determinant_non_square_matrix() {
        let matrix = Matrix {
            data: vec![1, 2, 3, 4, 5, 6],
            row_size: 2,
            col_size: 3,
        };
        assert_eq!(matrix.determinant(), None);
    }

    #[test]
    fn test_frobenius_norm_i32() {
        let matrix = Matrix::<i32> {
            data: vec![1, 2, 3, 4],
            row_size: 2,
            col_size: 2,
        };

        let expected: f64 = 5.477225575051661;
        let result = matrix.frobenius_norm();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_frobenius_norm_f32() {
        let matrix = Matrix::<f32> {
            data: vec![1.0, 2.0, 3.0, 4.0],
            row_size: 2,
            col_size: 2,
        };

        let expected: f64 = 5.477225575051661;
        let result = matrix.frobenius_norm();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_2x2() {
        let matrix = Matrix {
            data: vec![4.0, 7.0, 2.0, 6.0],
            row_size: 2,
            col_size: 2,
        };

        let expected_inverse = vec![0.6, -0.7, -0.2, 0.4];
        let result = matrix.inverse().unwrap();

        assert!(approx_equal(&result.data, &expected_inverse, 1e-6));
    }

    #[test]
    fn test_inverse_3x3() {
        let matrix = Matrix {
            data: vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0],
            row_size: 3,
            col_size: 3,
        };

        let expected_inverse = vec![-24.0, 18.0, 5.0, 20.0, -15.0, -4.0, -5.0, 4.0, 1.0];
        let result = matrix.inverse().unwrap();

        assert!(approx_equal(&result.data, &expected_inverse, 1e-6));
    }

    #[test]
    fn test_inverse_identity() {
        let matrix = Matrix::<f64>::identity(3);
        let result = matrix.inverse().unwrap();

        assert_eq!(result.data, matrix.data);
    }

    #[test]
    fn test_inverse_singular() {
        let matrix = Matrix {
            data: vec![1.0, 2.0, 2.0, 4.0],
            row_size: 2,
            col_size: 2,
        };

        let result = matrix.inverse();

        assert!(result.is_none());
    }

    #[test]
    fn test_inverse_1x1() {
        let matrix = Matrix {
            data: vec![2.0],
            row_size: 1,
            col_size: 1,
        };

        let expected_inverse = vec![0.5];
        let result = matrix.inverse().unwrap();

        assert!(approx_equal(&result.data, &expected_inverse, 1e-6));
    }
}
