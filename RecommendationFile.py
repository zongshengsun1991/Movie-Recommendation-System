# Movie Recommendation Systems
# Ryan Wedoff & Zongsheng Sun
# Data From http://grouplens.org/datasets/movielens/
import scipy.sparse
from scipy.sparse import linalg
import numpy as np

f = open('ml-100k/u.data', 'r', encoding='utf-8')
line = f.readline()

# chunk[0] = user id
# chunk[1] = item id
# chunk[2] = rating
# chunk[3] = timestamp
# print(chunk[0], chunk[1], chunk[2], chunk[3])

line_num = 1

rowList = []
colList = []
valList = []

for line in f:
    chunk = line.split("\t")
    rowList.append(float(chunk[0]))
    colList.append(float(chunk[1]))
    valList.append(float(chunk[2]))
f.close()

# create sparse matrices

sparseMatrix = scipy.sparse.csr_matrix((valList, (rowList, colList)))

# Average rating per movie code  we might have to strip blanks or something, this seems low
averageRating = sparseMatrix.mean(axis=0)
print(averageRating.max())  # Max rating
print(np.argsort(averageRating)[-1])  # This ranks movie ids from worst to best
print(averageRating)
print(averageRating.shape)

U, singValues, Vt = scipy.sparse.linalg.svds(sparseMatrix, k=10)
# print(singValues)


# Convert the sinular values to a diagonal matrix shape
Sigma = np.diag(singValues)

print(U[2].shape)
print(U.shape)
print(singValues.shape)
print(Sigma.shape)
print(Vt.shape)
print(U[1])

# This is the row/user we will choose
chosenRow = 0

# Multiply row 2 or left singular vectors, which is a song, and multiply by the diagonal
# of the singular values times the transpose of the riht singular values
relArray = U[chosenRow].dot(Sigma).dot(Vt)
# Sort the values so that the least worst movie is first, and the best used movie is last.
# Values is a sorted array of the indexes, sorting low to high
values = np.argsort(relArray)
print(values)

# NEED TO SUBTRACT THE MOVIES THAT THE PERSON HAS ALREADY SEEN

# Save and print the 10 most closely associated songs for word i = row 1
end = np.matrix(
        [values[-1], values[-2], values[-3], values[-4], values[-5], values[-6], values[-7], values[-8], values[-9],
         values[-10]])
print(end)
