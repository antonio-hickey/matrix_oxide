use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

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

    /// Perform a transpose operation (swap rows for columns and vice versa)
    /// Example:
    ///  [[1, 2, 3]       [[1, 4]
    ///   [4, 5, 6]]   ->  [2, 5]
    ///                    [3, 6]]
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
    /// Example:
    ///  [[1, 2, 3]     [[6, 5, 4],    [[-5, -3, -1]
    ///   [4, 5, 6]]  -  [3, 2, 1]]  =  [1, 3, 5]]
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
    T: Add<Output = T> + Mul<Output = T> + Clone,
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
