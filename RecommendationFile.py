# Movie Recommendation Systems
# Ryan Wedoff & Zongsheng Sun
# Data From http://grouplens.org/datasets/movielens/
import scipy.sparse
from scipy.sparse import linalg
import numpy as np
from flask import Flask, render_template, request, flash

app = Flask(__name__)
app.debug = True


@app.route("/")
def main():
    # noinspection PyUnresolvedReferences
    return render_template('index.html')


@app.route('/', methods=['POST'])
def my_form_post():
    recsVar = recommend(request.form['text'])
    # noinspection PyUnresolvedReferences
    return render_template('recommend.html', recs=recsVar)


def recommend(userid):
    f = open('ml-100k/u.data', 'r', encoding='utf-8')
    line = f.readline()

    # chunk[0] = user id
    # chunk[1] = item id
    # chunk[2] = rating
    # chunk[3] = timestamp

    rowlist = []
    collist = []
    vallist = []

    for line in f:
        chunk = line.split("\t")
        rowlist.append(int(chunk[0]) - 1)  # as the index starts from 0
        collist.append(int(chunk[1]) - 1)  # as the index starts from 0
        vallist.append(float(chunk[2]))
    f.close()

    # create sparse matrices
    sparseMatrix = scipy.sparse.csr_matrix((vallist, (rowlist, collist)))
    # Average rating per movie code  we might have to strip blanks or something, this seems low

    # calculate the average of nonzero element
    averageRating = np.multiply(sparseMatrix.sum(axis=0),
                                1.0 / (sparseMatrix != 0).sum(axis=0))  # calculate the average of nonzero element
    # calculate the Exceed Average matrix

    sparseMatrix -= scipy.sparse.csr_matrix((sparseMatrix != 0).multiply(averageRating))

    # get the average exceed average rate for each user
    averageExceedRating = np.multiply(sparseMatrix.sum(axis=1), 1.0 / (sparseMatrix != 0).sum(axis=1))

    # get the r matrix
    sparseMatrix -= scipy.sparse.csr_matrix((sparseMatrix != 0).multiply(averageExceedRating))

    # print(averageRating.max())  # Max rating
    # print(np.argsort(averageRating)[-1])  # This ranks movie ids from worst to best
    # print(averageRating)
    # print(averageRating.shape)

    U, singValues, Vt = scipy.sparse.linalg.svds(sparseMatrix,
                                                 k=300)  # k= 10 is too small, this is a question we need to ask
    # print(singValues)


    # Convert the sinular values to a diagonal matrix shape
    Sigma = np.diag(singValues)


    # This is the row/user we will choose

    userCount = sparseMatrix.get_shape()[0]
    chosenRow = int(userid)
    #todo the index might be off by 1!
    #todo save the matrix and pull it in so it isn't slow

    #if not userCount >= chosenRow >= 1:
    if not userCount > chosenRow >= 1:
        return "User Not Found"

    # Multiply row 2 or left singular vectors, which is a song, and multiply by the diagonal
    # of the singular values times the transpose of the riht singular values

    # add back the average rate and the AverageExceed Rate
    relArray = U[chosenRow].dot(Sigma).dot(Vt) + averageRating + averageExceedRating.item(chosenRow)

    # set all the already rated movie to 0
    relArray -= (sparseMatrix.getrow(chosenRow) != 0).multiply(relArray)
    relArray = np.asarray(relArray).flatten()

    # Sort the values so that the least worst movie is first, and the best used movie is last.
    # Values is a sorted array of the indexes, sorting low to high
    values = np.argsort(relArray)

    # Save and print the 10 most closely associated songs for word i = row 1
    end = np.array(
        [values[-1], values[-2], values[-3], values[-4], values[-5], values[-6], values[-7], values[-8], values[-9],
         values[-10]]) + 1  # just for the index start from 0
    print(end)

    moviesFile = open('ml-100k/u.item', 'r', encoding = "ISO-8859-1")
    dict = {}
    for movieLine in moviesFile:
        chunk = movieLine.split("|")
        dict[str(chunk[0])] = chunk[1]
    moviesFile.close()
    recommendMovies = []
    print(dict[str(end[0])])
    for movieId in end:
        recommendMovies.append(dict[str(movieId)])
    return recommendMovies


if __name__ == "__main__":
    app.run()
