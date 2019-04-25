import codecs
from math import sqrt
import numpy as np



class Recommender:
    def __init__(self,path):
        self.username2id = {}
        self.userid2name = {}
        self.productid2name = {}
        self.loadMovieLens(path)
        
        self.similarities = self.computeSimilarities(list(self.productid2name.keys()), self.data)
        self.allPredictions()
        self.printPredictions()

    def convertProductID2name(self, id):
        """Given product id number return product name"""
        if id in self.productid2name:
            return self.productid2name[id]
        else:
            return id

    def loadMovieLens(self, path=''):
        self.data = {}
        #
        # first load movie ratings
        #
        i = 0
        #
        # First load book ratings into self.data
        #
        #f = codecs.open(path + "u.data", 'r', 'utf8')
        f = codecs.open(path + "u.data", 'r', 'ascii')
        #  f = open(path + "u.data")
        for line in f:
            i += 1
            #separate line into fields
            fields = line.split('\t')
            user = fields[0]
            movie = fields[1]
            rating = int(fields[2].strip().strip('"'))
            if user in self.data:
                currentRatings = self.data[user]
            else:
                currentRatings = {}
            currentRatings[movie] = rating
            self.data[user] = currentRatings
        f.close()
        #
        # Now load movie into self.productid2name
        # the file u.item contains movie id, title, release date among
        # other fields
        #
        #f = codecs.open(path + "u.item", 'r', 'utf8')
        f = codecs.open(path + "u.item", 'r', 'iso8859-1', 'ignore')
        #f = open(path + "u.item")
        for line in f:
            i += 1
            #separate line into fields
            fields = line.split('|')
            mid = fields[0].strip()
            title = fields[1].strip()
            self.productid2name[mid] = title
        f.close()
        #
        #  Now load user info into both self.userid2name
        #  and self.username2id
        #
        #f = codecs.open(path + "u.user", 'r', 'utf8')
        f = open(path + "u.user")
        for line in f:
            i += 1
            fields = line.split('|')
            userid = fields[0].strip('"')
            self.userid2name[userid] = line
            self.username2id[line] = userid
        f.close()
        print(i)


    def computeSimilarities(self,bands, userRatings):
        similarities = {}
        for i in bands:
            similarities.setdefault(i, {})
            for j in bands:
                similarity = self.computeSimilarity(i, j, userRatings)
                if(similarity != -2):
                    similarities[i][j] = similarity
        return similarities

    def computeUserAverages(self,users):
        results = {}
        for (key, ratings) in users.items():
            results[key] = float(sum(ratings.values())) / len(ratings.values())
        return results

    def computeSimilarity(self,band1, band2, userRatings):
        averages = {}
        for (key, ratings) in userRatings.items():
            averages[key] = (float(sum(ratings.values()))
                            / len(ratings.values()))

        num  = 0  # numerator
        dem1 = 0 # first half of denominator
        dem2 = 0
        enters = False
        for (user, ratings) in userRatings.items():
            if band1 in ratings and band2 in ratings:
                avg = averages[user]
                num += (ratings[band1] - avg) * (ratings[band2] - avg)
                dem1 += (ratings[band1] - avg)**2
                dem2 += (ratings[band2] - avg)**2
                enters = True
  
        if(enters):
            return num / (sqrt(dem1) * sqrt(dem2))
        else:
            return -2

    def normalizeRating(self,rating, minR, maxR):
        return (2*(rating - minR) - (maxR - minR)) / (maxR - minR)

    def denormalizeRating(self,norm_rating, minR, maxR):
        return 0.5*((norm_rating + 1) * (maxR - minR)) + minR

    def predictRating(self,user, band):
        
        num = 0.0
        dem = 0.0
        for N in self.similarities:
            if N in self.similarities[band] and N in self.users[user]:
                num += self.similarities[band][N] * self.normalizeRating(self.users[user][N], 1, 5)
                dem += abs(self.similarities[band][N])

        rating = self.denormalizeRating(num/dem, 1, 5)
        return rating
        

    def allPredictions(self):
        self.ratings = {}
        for (k, i) in self.users.items():
            self.ratings.setdefault(k, {})
            for j in self.bands:
                if not j in i:
                    self.ratings[k][j] = self.predictRating(k,j)

    def printPredictions(self):
        for k in self.ratings.keys():
            print("Predicciones para ",k,":\n")
            b = list(self.ratings[k].items())
            b.sort(key=lambda x:x[1])
            b.reverse()
            if(len(b) > 0):
                for k,v in b:
                    print("%20s%10.5f" % (k,v)  )
            else:
                print("Ya calificó a todos, no hay predicción.")
            print("\n")

        
        

def main():
    r = Recommender("/Users/lamlemonpie/Documents/CS-UNSA/2019-A/TBD/BDS/ml-100k/")



if __name__ == "__main__":
    main()
