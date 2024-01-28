use std::fmt::Debug;
use std::ops::Add;

/// Non resizeable MxN Matrix
pub struct Matrix<T, const R: usize, const C: usize> {
    pub data: Vec<T>,
}
impl<T: Default + Clone, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Construct a new *non-empty* and *sized* `Matrix`
    pub fn new() -> Self {
        Matrix {
            data: vec![T::default(); R * C],
        }
    }

    /// Try to get a reference to the value at a given row and column from the matrix
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < R && col < C {
            Some(&self.data[col + row * C])
        } else {
            None
        }
    }

    /// Try to get a mutable reference to the value at a given row and column from the matrix
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < R && col < C {
            Some(&mut self.data[col + row * C])
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
        if column >= C {
            return None;
        }

        // Iterate over all the rows grabbing a specific column each time
        let col_data: Vec<T> = (0..R)
            .map(|row| self.data[row * C + column].clone())
            .collect();

        Some(col_data)
    }

    /// Try to get all the values for a given row
    ///
    /// NOTE: If you pass a row value larger than the number of rows
    /// this function will return None.
    pub fn try_get_row(&self, row: usize) -> Option<Vec<T>> {
        // Bounds check
        if row >= R {
            return None;
        }

        // Iterate over all the rows grabbing a specific column each time
        let row_data: Vec<T> = (0..C).map(|col| self.data[row * C + col].clone()).collect();

        Some(row_data)
    }

    /// Perform a transpose operation (swap rows for columns and vice versa)
    /// Example:
    ///  [[1, 2, 3]       [[1, 4]
    ///   [4, 5, 6]]   ->  [2, 5]
    ///                    [3, 6]]
    pub fn transpose(&self) -> Matrix<T, R, C> {
        Matrix {
            data: (0..C)
                .flat_map(|col| (0..R).map(move |row| self.data[row * C + col].clone()))
                .collect(),
        }
    }
}
impl<T: Default + Clone, const R: usize, const C: usize> Default for Matrix<T, R, C> {
    /// Create a default `Matrix` instance
    fn default() -> Self {
        Self::new()
    }
}
impl<T: Default + Clone + Debug, const R: usize, const C: usize> Add for Matrix<T, R, C>
where
    T: Add<Output = T> + Clone,
{
    type Output = Matrix<T, R, C>;

    /// Matrix addition
    /// NOTE: the matrices you add MUST have the same dimensionality
    fn add(self, rhs: Self) -> Matrix<T, R, C> {
        let data: Vec<T> = (0..R)
            .flat_map(|row| {
                let row_a = self.try_get_row(row).expect("Invalid row in self");
                let row_b = rhs.try_get_row(row).expect("Invalid row in rhs");
                row_a.into_iter().zip(row_b).map(|(a, b)| a + b)
            })
            .collect();

        Matrix { data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_addition() {
        // Create two 2x2 matrices
        let matrix_a = Matrix::<i32, 2, 2> {
            data: vec![1, 2, 3, 4],
        };
        let matrix_b = Matrix::<i32, 2, 2> {
            data: vec![4, 3, 2, 1],
        };

        // Expected result of addition
        let expected = Matrix::<i32, 2, 2> {
            data: vec![5, 5, 5, 5],
        };

        // Perform addition
        let result = matrix_a.add(matrix_b);

        // Assert that the result is as expected
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn new_matrix_has_correct_size() {
        let matrix: Matrix<i32, 2, 3> = Matrix::new();
        assert_eq!(matrix.data.len(), 6);
    }

    #[test]
    fn default_matrix_is_same_as_new() {
        let default_matrix: Matrix<i32, 2, 3> = Matrix::default();
        let new_matrix: Matrix<i32, 2, 3> = Matrix::new();
        assert_eq!(default_matrix.data, new_matrix.data);
    }

    #[test]
    fn try_get_column_valid() {
        let matrix: Matrix<i32, 2, 3> = Matrix::new();
        let column = matrix.try_get_column(1);
        assert!(column.is_some());
        assert_eq!(column.unwrap(), vec![0, 0]);
    }

    #[test]
    fn try_get_column_invalid() {
        let matrix: Matrix<i32, 2, 3> = Matrix::new();
        let column = matrix.try_get_column(3);
        assert!(column.is_none());
    }

    #[test]
    fn try_get_row_valid() {
        let matrix: Matrix<i32, 2, 3> = Matrix::new();
        let row = matrix.try_get_row(0);
        assert!(row.is_some());
        assert_eq!(row.unwrap(), vec![0, 0, 0]);
    }

    #[test]
    fn try_get_row_invalid() {
        let matrix: Matrix<i32, 2, 3> = Matrix::new();
        let row = matrix.try_get_row(2);
        assert!(row.is_none());
    }

    #[test]
    fn transpose_works_correctly() {
        let mut matrix: Matrix<i32, 2, 3> = Matrix::new();
        for i in 0..matrix.data.len() {
            matrix.data[i] = i as i32;
        }
        let transposed = matrix.transpose();
        assert_eq!(transposed.data, vec![0, 3, 1, 4, 2, 5]);
    }
}
