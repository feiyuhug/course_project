import feedparser
from bayes import *

#ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
#sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')

#getTopWords(ny, sf)
#vocabList, p0V, p1V = localWords(ny, sf)
frequentWords = feedparser.parse('http://www.ranks.nl/resources/stopwords.html')
print len(frequentWords['entries'])
for i in range(len(frequentWords['entries'])) :
    print frequentWords['entries'][i]

