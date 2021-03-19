#!/usr/bin/env python3
import numpy as np

# NOTE: It would be probably faster and certainly more versatile to use SciPy implementation
# of sparse matrices but since we had to implement document vector representation, 
# this is the option I chose.

class BaseSparseMatrix:
    """ Base class for sparse matrices

    Attributes:
        M: number of rows
        N: number of columns
        data: list of elements
        indices: indices in rows/columns
        indptr: indptr[i] number of nonzero elements in previous rows/cols

    """
    def __init__(self, matrix, shape, dtype):
        self._create(matrix, shape, dtype)

    def _create(self, matrix, shape, dtype):
        """ Create a sparse (CSR or CSC) matrix

        `indices1` and `indices2` are row and column indices for CSR matrix
        and swaped for CSC matrix (ie. the matrix is transposed during creation
        of its sparse variant)

        Args:
            dtype: type of data
            matrix: (data, (indices1, indices2))
            shape: shape of dense matrix

        """
        (data, (row, col)) = matrix

        (self.M, self.N) = shape
        self.indptr[0] = 0
        self.data = np.empty(len(data))
        self.indices = np.empty(len(col), dtype=np.int32)

        row_sort_ind = np.argsort(row)

        row, col, data = row[row_sort_ind], col[row_sort_ind], data[row_sort_ind]

        i = 0
        nnz = 0
        end_sub=start_of_row=end_of_row=0

        while end_of_row < len(row):
            start_of_row = end_of_row
            # find index, on which the next row in matrix starts
            # ie. data[:end_sub] containes the current and previoues rows
            end_sub = np.searchsorted(row[start_of_row:], row[start_of_row]+1, sorter=None)
            end_of_row += end_sub
            size_of_row = end_of_row - start_of_row

            cols_in_row = col[start_of_row:end_of_row]
            col_sorted_indices = np.argsort(cols_in_row)

            cols_in_row = cols_in_row[col_sorted_indices]
            data_in_row = data[start_of_row:end_of_row][col_sorted_indices]
            for col2, data_point in zip(cols_in_row, data_in_row):
                self.data[i] = data_point
                self.indices[i] = col2
                i += 1
            prev_nnz = nnz
            nnz += len(data_in_row)

            # add indptr for empty rows (=prev_nnz)
            self.indptr[row[start_of_row-1]+1:row[start_of_row]+1] = prev_nnz

            # add NNZ for this row at +1 position
            self.indptr[row[start_of_row]+1] = nnz

        # there may be empty rows after last data_point
        self.indptr[row[-1]+1:self.M+1] = nnz


    def sequence_iterator(self):
        """ Iterate over rows (in CSR matrix) or columns (in CSC matrix)

        Yields:
            tuple (indices, data) of row/column

        """
        #for each row/col yield (indiceS, data_pointS)
        for i in range(len(self.indptr)-1):
            s, e = self.indptr[i], self.indptr[i+1]
            yield (self.indices[s:e], self.data[s:e])

    @property
    def shape(self):
        return (self.M, self.N)


class csr_matrix(BaseSparseMatrix):
    """ Sparse Matrix in Compressed Row Rormat
    """
    def __init__(self, matrix, shape, dtype=np.float32):
        """ Create matrix in CSR format

        Args:
            matrix: ((data, (row_ind, col_ind))
            shape: shape of dense matrix
            dtype: (optional) type of data

        """
        self.indptr = np.empty(shape[0]+1, dtype=np.int32)
        super().__init__(matrix, shape, dtype)


    def dot(self, other):
        """ Perform dot product with matrix in CSC format

        Args:
            other: matrix in CSC format.

        Returns:
            Dense matrix (np.array) with the resutl

        """
        assert isinstance(other, csc_matrix)
        assert self.N == other.M
        # self = left, other = right
        dot = np.zeros((self.M, other.N), dtype=np.float64)
        self_iter = self.sequence_iterator()
        for row, (self_ind, self_data) in enumerate(self_iter):
            other_iter = other.sequence_iterator()
            for col, (other_ind, other_data) in enumerate(other_iter):
                i=j=0  # index_pointer
                result = 0
                while i < len(self_ind) and j < len(other_ind):
                    if self_ind[i] == other_ind[j]:
                        result += np.multiply(self_data[i], other_data[j], dtype=np.float64)
                        i+=1
                        j+=1
                    elif self_ind[i] < other_ind[j]:
                        i += 1
                    else:
                        j += 1
                dot[row, col] = result
        return dot


class csc_matrix(BaseSparseMatrix):
    """ Sparse Matrix in Compressed Column format
    """
    def __init__(self, matrix, shape, dtype=np.float32):
        """ Create matrix in CSC format

        Args:
            matrix: ((data, (row_ind, col_ind))
            shape: shape of dense matrix
            dtype: (optional) type of data

        """
        (data, (row, col)) = matrix
        self.indptr = np.empty(shape[1]+1, dtype=np.int32)
        super().__init__((data, (col, row)), shape, dtype)


