import codecs
from math import sqrt
import numpy as np

users2 = {"Amy": {"Taylor Swift": 4, "PSY": 3, "Whitney Houston": 4},
          "Ben": {"Taylor Swift": 5, "PSY": 2},
          "Clara": {"PSY": 3.5, "Whitney Houston": 4},
          "Daisy": {"Taylor Swift": 5, "Whitney Houston": 3}}

users = {"Angelica": {"Blues Traveler": 3.5, "Broken Bells": 2.0,
                      "Norah Jones": 4.5, "Phoenix": 5.0,
                      "Slightly Stoopid": 1.5, "The Strokes": 2.5,
                      "Vampire Weekend": 2.0},
         "Bill":{"Blues Traveler": 2.0, "Broken Bells": 3.5,
                 "Deadmau5": 4.0, "Phoenix": 2.0,
                 "Slightly Stoopid": 3.5, "Vampire Weekend": 3.0},
         "Chan": {"Blues Traveler": 5.0, "Broken Bells": 1.0,
                  "Deadmau5": 1.0, "Norah Jones": 3.0,
                  "Phoenix": 5, "Slightly Stoopid": 1.0},
         "Dan": {"Blues Traveler": 3.0, "Broken Bells": 4.0,
                 "Deadmau5": 4.5, "Phoenix": 3.0,
                 "Slightly Stoopid": 4.5, "The Strokes": 4.0,
                 "Vampire Weekend": 2.0},
         "Hailey": {"Broken Bells": 4.0, "Deadmau5": 1.0,
                    "Norah Jones": 4.0, "The Strokes": 4.0,
                    "Vampire Weekend": 1.0},
         "Jordyn":  {"Broken Bells": 4.5, "Deadmau5": 4.0,
                     "Norah Jones": 5.0, "Phoenix": 5.0,
                     "Slightly Stoopid": 4.5, "The Strokes": 4.0,
                     "Vampire Weekend": 4.0},
         "Sam": {"Blues Traveler": 5.0, "Broken Bells": 2.0,
                 "Norah Jones": 3.0, "Phoenix": 5.0,
                 "Slightly Stoopid": 4.0, "The Strokes": 5.0},
         "Veronica": {"Blues Traveler": 3.0, "Norah Jones": 5.0,
                      "Phoenix": 4.0, "Slightly Stoopid": 2.5,
                      "The Strokes": 3.0}
        }

users3 = {"David": {"Imagine Dragons": 3, "Daft Punk": 5,
                    "Lorde": 4, "Fall Out Boy": 1},
          "Matt":  {"Imagine Dragons": 3, "Daft Punk": 4,
                    "Lorde": 4, "Fall Out Boy": 1},
          "Ben":   {"Kacey Musgraves": 4, "Imagine Dragons": 3,
                    "Lorde": 3, "Fall Out Boy": 1},
          "Chris": {"Kacey Musgraves": 4, "Imagine Dragons": 4,
                    "Daft Punk": 4, "Lorde": 3, "Fall Out Boy": 1},
          "Tori":  {"Kacey Musgraves": 5, "Imagine Dragons": 4,
                    "Daft Punk": 5, "Fall Out Boy": 3}}


class Recommender:
    def __init__(self,users):
        self.users  = users
        self.rating = None
        self.bands  = None
        
        self.getBands()
        print("Bandas:",self.bands)
        self.similarities = self.computeSimilarities(self.bands, self.users)
        self.allPredictions()
        self.printPredictions()



    def getBands(self):
        bands = []
        for (k,v) in self.users.items():
            bands.extend(list(v.keys()))

        self.bands = list(set(bands))

    def computeSimilarities(self,bands, userRatings):
        similarities = {}
        for i in bands:
            similarities.setdefault(i, {})
            for j in bands:
                similarities[i][j] = self.computeSimilarity(i, j, userRatings)
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
        for (user, ratings) in userRatings.items():
            if band1 in ratings and band2 in ratings:
                avg = averages[user]
                num += (ratings[band1] - avg) * (ratings[band2] - avg)
                dem1 += (ratings[band1] - avg)**2
                dem2 += (ratings[band2] - avg)**2
        return num / (sqrt(dem1) * sqrt(dem2))

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
    r = Recommender(users3)



if __name__ == "__main__":
    main()
