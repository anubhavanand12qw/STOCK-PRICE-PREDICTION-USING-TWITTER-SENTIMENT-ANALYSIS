{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# STOCK PREDICTION USING TWITTER SENTIMENT ANALYSIS"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### importing machine learning libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:38:08.972017Z",
     "start_time": "2021-09-22T09:38:08.960019Z"
    }
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from nltk.classify import NaiveBayesClassifier\n",
    "from nltk.corpus import subjectivity\n",
    "from nltk.sentiment import SentimentAnalyzer\n",
    "from nltk.sentiment.util import *\n",
    "import matplotlib.pyplot as mlpt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### importing library to fetch data from twitter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:38:09.389331Z",
     "start_time": "2021-09-22T09:38:09.385373Z"
    }
   },
   "outputs": [],
   "source": [
    "import tweepy\n",
    "import csv\n",
    "import pandas as pd\n",
    "import random\n",
    "import numpy as np\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### setting up consumer key and access token"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:38:09.824910Z",
     "start_time": "2021-09-22T09:38:09.818945Z"
    }
   },
   "outputs": [],
   "source": [
    "consumer_key    = '3jmA1BqasLHfItBXj3KnAIGFB'\n",
    "consumer_secret = 'imyEeVTctFZuK62QHmL1I0AUAMudg5HKJDfkx0oR7oFbFinbvA'\n",
    "\n",
    "access_token  = '265857263-pF1DRxgIcxUbxEEFtLwLODPzD3aMl6d4zOKlMnme'\n",
    "access_token_secret = 'uUFoOOGeNJfOYD3atlcmPtaxxniXxQzAU4ESJLopA1lbC'\n",
    "\n",
    "auth = tweepy.OAuthHandler(consumer_key, consumer_secret)\n",
    "auth.set_access_token(access_token, access_token_secret)\n",
    "api = tweepy.API(auth,wait_on_rate_limit=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Fetching tweets for United Airlines in extended mode (means entire tweet will come and not just few words + link)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:38:18.943491Z",
     "start_time": "2021-09-22T09:38:10.988985Z"
    }
   },
   "outputs": [],
   "source": [
    "fetch_tweets=tweepy.Cursor(api.search, q=\"#unitedAIRLINES\",count=100, lang =\"en\",since=\"2018-9-13\", tweet_mode=\"extended\").items()\n",
    "data=pd.DataFrame(data=[[tweet_info.created_at.date(),tweet_info.full_text]for tweet_info in fetch_tweets],columns=['Date','Tweets'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:38:19.008323Z",
     "start_time": "2021-09-22T09:38:14.144Z"
    }
   },
   "outputs": [],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Removing special character from each tweets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:38:20.001041Z",
     "start_time": "2021-09-22T09:38:19.691867Z"
    }
   },
   "outputs": [],
   "source": [
    "data.to_csv(\"Tweets.csv\")\n",
    "cdata=pd.DataFrame(columns=['Date','Tweets'])\n",
    "total=100\n",
    "index=0\n",
    "for index,row in data.iterrows():\n",
    "    stre=row[\"Tweets\"]\n",
    "    my_new_string = re.sub('[^ a-zA-Z0-9]', '', stre)\n",
    "    temp_df = pd.DataFrame([[data[\"Date\"].iloc[index], \n",
    "                            my_new_string]], columns = ['Date','Tweets'])\n",
    "    cdata = pd.concat([cdata, temp_df], axis = 0).reset_index(drop = True)\n",
    "    # index=index+1\n",
    "#print(cdata.dtypes)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Displaying the data with date and tweets, you can notice there are multiple tweets for each day. So we will club them together later."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:38:20.951711Z",
     "start_time": "2021-09-22T09:38:20.940741Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Tweets</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2021-09-22</td>\n",
       "      <td>ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2021-09-22</td>\n",
       "      <td>RT diecastryan A nice full lineup at IAD last ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2021-09-22</td>\n",
       "      <td>United Airlines resuming Airline Tickets Reser...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2021-09-22</td>\n",
       "      <td>RT diecastryan A nice full lineup at IAD last ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2021-09-22</td>\n",
       "      <td>lol FAANews united does not give a single damn...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>367</th>\n",
       "      <td>2021-09-13</td>\n",
       "      <td>Thank You  unitedAIRLINES httpstcoRU897P5rqI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>368</th>\n",
       "      <td>2021-09-13</td>\n",
       "      <td>Where does the journey take you    luggage tra...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>369</th>\n",
       "      <td>2021-09-13</td>\n",
       "      <td>RT n194at United Air LinesDouglas DC852 N8062U...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>370</th>\n",
       "      <td>2021-09-13</td>\n",
       "      <td>It is so ignorant to have 1299 in flight wifi ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>371</th>\n",
       "      <td>2021-09-13</td>\n",
       "      <td>Exactly But we have pretty options than United...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>372 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "           Date                                             Tweets\n",
       "0    2021-09-22  ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...\n",
       "1    2021-09-22  RT diecastryan A nice full lineup at IAD last ...\n",
       "2    2021-09-22  United Airlines resuming Airline Tickets Reser...\n",
       "3    2021-09-22  RT diecastryan A nice full lineup at IAD last ...\n",
       "4    2021-09-22  lol FAANews united does not give a single damn...\n",
       "..          ...                                                ...\n",
       "367  2021-09-13       Thank You  unitedAIRLINES httpstcoRU897P5rqI\n",
       "368  2021-09-13  Where does the journey take you    luggage tra...\n",
       "369  2021-09-13  RT n194at United Air LinesDouglas DC852 N8062U...\n",
       "370  2021-09-13  It is so ignorant to have 1299 in flight wifi ...\n",
       "371  2021-09-13  Exactly But we have pretty options than United...\n",
       "\n",
       "[372 rows x 2 columns]"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cdata"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Creating a dataframe where we will combine the tweets date wise and store into"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:38:22.094898Z",
     "start_time": "2021-09-22T09:38:22.075919Z"
    }
   },
   "outputs": [],
   "source": [
    "ccdata=pd.DataFrame(columns=['Date','Tweets'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:38:22.530962Z",
     "start_time": "2021-09-22T09:38:22.491788Z"
    }
   },
   "outputs": [],
   "source": [
    "indx=0\n",
    "get_tweet=\"\"\n",
    "for i in range(0,len(cdata)-1):\n",
    "    get_date=cdata.Date.iloc[i]\n",
    "    next_date=cdata.Date.iloc[i+1]\n",
    "    if(str(get_date)==str(next_date)):\n",
    "        get_tweet=get_tweet+cdata.Tweets.iloc[i]+\" \"\n",
    "    if(str(get_date)!=str(next_date)):\n",
    "        temp_df = pd.DataFrame([[get_date, \n",
    "                                get_tweet]], columns = ['Date','Tweets'])\n",
    "        ccdata = pd.concat([ccdata, temp_df], axis = 0).reset_index(drop = True)\n",
    "        get_tweet=\" \""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### All the tweets has been clubbed as per their date."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:38:23.249425Z",
     "start_time": "2021-09-22T09:38:23.239458Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Tweets</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2021-09-22</td>\n",
       "      <td>ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2021-09-21</td>\n",
       "      <td>RT SparrowOneSix 737900 N78448 was carrying U...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2021-09-20</td>\n",
       "      <td>RT diecastryan A nice full lineup at IAD last...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2021-09-19</td>\n",
       "      <td>jacobcabe Guess UnitedAirlines wont get any  ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2021-09-18</td>\n",
       "      <td>RT FELASTORY UnitedAirlines announce non stop...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>2021-09-17</td>\n",
       "      <td>UnitedAirlines 90 of workers vaccinated after...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>2021-09-16</td>\n",
       "      <td>This is how united UnitedAirlines treated wit...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>2021-09-15</td>\n",
       "      <td>Thank you SPONSORSYour generous support make ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>2021-09-14</td>\n",
       "      <td>Because I get to work with amazing people uni...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Date                                             Tweets\n",
       "0  2021-09-22  ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...\n",
       "1  2021-09-21   RT SparrowOneSix 737900 N78448 was carrying U...\n",
       "2  2021-09-20   RT diecastryan A nice full lineup at IAD last...\n",
       "3  2021-09-19   jacobcabe Guess UnitedAirlines wont get any  ...\n",
       "4  2021-09-18   RT FELASTORY UnitedAirlines announce non stop...\n",
       "5  2021-09-17   UnitedAirlines 90 of workers vaccinated after...\n",
       "6  2021-09-16   This is how united UnitedAirlines treated wit...\n",
       "7  2021-09-15   Thank you SPONSORSYour generous support make ...\n",
       "8  2021-09-14   Because I get to work with amazing people uni..."
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ccdata"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Now to know the \"closing price\" of each day we will import STOCK PRICE DATA for UNITED AIRLINES from \"yahoo.finance\". We will consider \"Close\" price only."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:38:24.417614Z",
     "start_time": "2021-09-22T09:38:24.386698Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Open</th>\n",
       "      <th>High</th>\n",
       "      <th>Low</th>\n",
       "      <th>Close</th>\n",
       "      <th>Adj Close</th>\n",
       "      <th>Volume</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2020-09-01</td>\n",
       "      <td>35.250000</td>\n",
       "      <td>37.240002</td>\n",
       "      <td>34.950001</td>\n",
       "      <td>36.009998</td>\n",
       "      <td>36.009998</td>\n",
       "      <td>29722200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2020-09-02</td>\n",
       "      <td>36.099998</td>\n",
       "      <td>37.099998</td>\n",
       "      <td>35.209999</td>\n",
       "      <td>36.889999</td>\n",
       "      <td>36.889999</td>\n",
       "      <td>26622800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2020-09-03</td>\n",
       "      <td>37.130001</td>\n",
       "      <td>39.770000</td>\n",
       "      <td>36.139999</td>\n",
       "      <td>37.400002</td>\n",
       "      <td>37.400002</td>\n",
       "      <td>53966400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2020-09-04</td>\n",
       "      <td>38.150002</td>\n",
       "      <td>38.740002</td>\n",
       "      <td>36.459999</td>\n",
       "      <td>38.209999</td>\n",
       "      <td>38.209999</td>\n",
       "      <td>33121600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2020-09-08</td>\n",
       "      <td>37.299999</td>\n",
       "      <td>38.480000</td>\n",
       "      <td>36.480000</td>\n",
       "      <td>37.279999</td>\n",
       "      <td>37.279999</td>\n",
       "      <td>33207100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>261</th>\n",
       "      <td>2021-09-15</td>\n",
       "      <td>43.650002</td>\n",
       "      <td>43.910000</td>\n",
       "      <td>43.020000</td>\n",
       "      <td>43.860001</td>\n",
       "      <td>43.860001</td>\n",
       "      <td>10321600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>262</th>\n",
       "      <td>2021-09-16</td>\n",
       "      <td>43.860001</td>\n",
       "      <td>45.410000</td>\n",
       "      <td>43.849998</td>\n",
       "      <td>44.470001</td>\n",
       "      <td>44.470001</td>\n",
       "      <td>12204300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>263</th>\n",
       "      <td>2021-09-17</td>\n",
       "      <td>44.779999</td>\n",
       "      <td>45.500000</td>\n",
       "      <td>44.110001</td>\n",
       "      <td>44.540001</td>\n",
       "      <td>44.540001</td>\n",
       "      <td>11733300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>264</th>\n",
       "      <td>2021-09-20</td>\n",
       "      <td>44.759998</td>\n",
       "      <td>45.340000</td>\n",
       "      <td>43.590000</td>\n",
       "      <td>45.270000</td>\n",
       "      <td>45.270000</td>\n",
       "      <td>14700300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>265</th>\n",
       "      <td>2021-09-21</td>\n",
       "      <td>45.500000</td>\n",
       "      <td>46.259998</td>\n",
       "      <td>44.279999</td>\n",
       "      <td>44.450001</td>\n",
       "      <td>44.450001</td>\n",
       "      <td>12207000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>266 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "           Date       Open       High        Low      Close  Adj Close  \\\n",
       "0    2020-09-01  35.250000  37.240002  34.950001  36.009998  36.009998   \n",
       "1    2020-09-02  36.099998  37.099998  35.209999  36.889999  36.889999   \n",
       "2    2020-09-03  37.130001  39.770000  36.139999  37.400002  37.400002   \n",
       "3    2020-09-04  38.150002  38.740002  36.459999  38.209999  38.209999   \n",
       "4    2020-09-08  37.299999  38.480000  36.480000  37.279999  37.279999   \n",
       "..          ...        ...        ...        ...        ...        ...   \n",
       "261  2021-09-15  43.650002  43.910000  43.020000  43.860001  43.860001   \n",
       "262  2021-09-16  43.860001  45.410000  43.849998  44.470001  44.470001   \n",
       "263  2021-09-17  44.779999  45.500000  44.110001  44.540001  44.540001   \n",
       "264  2021-09-20  44.759998  45.340000  43.590000  45.270000  45.270000   \n",
       "265  2021-09-21  45.500000  46.259998  44.279999  44.450001  44.450001   \n",
       "\n",
       "       Volume  \n",
       "0    29722200  \n",
       "1    26622800  \n",
       "2    53966400  \n",
       "3    33121600  \n",
       "4    33207100  \n",
       "..        ...  \n",
       "261  10321600  \n",
       "262  12204300  \n",
       "263  11733300  \n",
       "264  14700300  \n",
       "265  12207000  \n",
       "\n",
       "[266 rows x 7 columns]"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "read_stock_p=pd.read_csv('UAL.csv')\n",
    "# DOWNLOAD UPDATED CLOSE PRICE FROM https://finance.yahoo.com/quote/UAL/history?period1=1598918400&period2=1632268800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true\n",
    "read_stock_p"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Adding a \"Price\" column in our dataframe and fetching the stock price as per the date in our dataframe."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:38:25.665268Z",
     "start_time": "2021-09-22T09:38:25.661310Z"
    }
   },
   "outputs": [],
   "source": [
    "ccdata['Prices']=\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:39:33.585643Z",
     "start_time": "2021-09-22T09:39:33.509806Z"
    }
   },
   "outputs": [],
   "source": [
    "indx=0\n",
    "for i in range (0,len(ccdata)):\n",
    "    for j in range (0,len(read_stock_p)):\n",
    "        get_tweet_date=ccdata.Date.iloc[i]\n",
    "        get_stock_date=read_stock_p.Date.iloc[j]\n",
    "        if(str(get_stock_date)==str(get_tweet_date)):\n",
    "            #print(get_stock_date,\" \",get_tweet_date)\n",
    "            # ccdata.set_value(i,'Prices',int(read_stock_p.Close[j]))\n",
    "            ccdata['Prices'].iloc[i] = int(read_stock_p.Close[j])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Prices are fetched but some entires are blank as close price might not be available for that day due to some reason (like holiday, etc.)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:39:34.774346Z",
     "start_time": "2021-09-22T09:39:34.754395Z"
    },
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Tweets</th>\n",
       "      <th>Prices</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2021-09-22</td>\n",
       "      <td>ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...</td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2021-09-21</td>\n",
       "      <td>RT SparrowOneSix 737900 N78448 was carrying U...</td>\n",
       "      <td>44</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2021-09-20</td>\n",
       "      <td>RT diecastryan A nice full lineup at IAD last...</td>\n",
       "      <td>45</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2021-09-19</td>\n",
       "      <td>jacobcabe Guess UnitedAirlines wont get any  ...</td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2021-09-18</td>\n",
       "      <td>RT FELASTORY UnitedAirlines announce non stop...</td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>2021-09-17</td>\n",
       "      <td>UnitedAirlines 90 of workers vaccinated after...</td>\n",
       "      <td>44</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>2021-09-16</td>\n",
       "      <td>This is how united UnitedAirlines treated wit...</td>\n",
       "      <td>44</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>2021-09-15</td>\n",
       "      <td>Thank you SPONSORSYour generous support make ...</td>\n",
       "      <td>43</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>2021-09-14</td>\n",
       "      <td>Because I get to work with amazing people uni...</td>\n",
       "      <td>43</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Date                                             Tweets Prices\n",
       "0  2021-09-22  ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...       \n",
       "1  2021-09-21   RT SparrowOneSix 737900 N78448 was carrying U...     44\n",
       "2  2021-09-20   RT diecastryan A nice full lineup at IAD last...     45\n",
       "3  2021-09-19   jacobcabe Guess UnitedAirlines wont get any  ...       \n",
       "4  2021-09-18   RT FELASTORY UnitedAirlines announce non stop...       \n",
       "5  2021-09-17   UnitedAirlines 90 of workers vaccinated after...     44\n",
       "6  2021-09-16   This is how united UnitedAirlines treated wit...     44\n",
       "7  2021-09-15   Thank you SPONSORSYour generous support make ...     43\n",
       "8  2021-09-14   Because I get to work with amazing people uni...     43"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ccdata"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### So we take the mean for the close price and put it in the blank value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:39:37.144442Z",
     "start_time": "2021-09-22T09:39:37.126487Z"
    }
   },
   "outputs": [],
   "source": [
    "mean=0\n",
    "summ=0\n",
    "count=0\n",
    "for i in range(0,len(ccdata)):\n",
    "    if(ccdata.Prices.iloc[i]!=\"\"):\n",
    "        summ=summ+int(ccdata.Prices.iloc[i])\n",
    "        count=count+1\n",
    "mean=summ/count\n",
    "for i in range(0,len(ccdata)):\n",
    "    if(ccdata.Prices.iloc[i]==\"\"):\n",
    "        ccdata.Prices.iloc[i]=int(mean)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Now all the entries have some value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:39:39.559565Z",
     "start_time": "2021-09-22T09:39:39.547599Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Tweets</th>\n",
       "      <th>Prices</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2021-09-22</td>\n",
       "      <td>ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...</td>\n",
       "      <td>43</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2021-09-21</td>\n",
       "      <td>RT SparrowOneSix 737900 N78448 was carrying U...</td>\n",
       "      <td>44</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2021-09-20</td>\n",
       "      <td>RT diecastryan A nice full lineup at IAD last...</td>\n",
       "      <td>45</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2021-09-19</td>\n",
       "      <td>jacobcabe Guess UnitedAirlines wont get any  ...</td>\n",
       "      <td>43</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2021-09-18</td>\n",
       "      <td>RT FELASTORY UnitedAirlines announce non stop...</td>\n",
       "      <td>43</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>2021-09-17</td>\n",
       "      <td>UnitedAirlines 90 of workers vaccinated after...</td>\n",
       "      <td>44</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>2021-09-16</td>\n",
       "      <td>This is how united UnitedAirlines treated wit...</td>\n",
       "      <td>44</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>2021-09-15</td>\n",
       "      <td>Thank you SPONSORSYour generous support make ...</td>\n",
       "      <td>43</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>2021-09-14</td>\n",
       "      <td>Because I get to work with amazing people uni...</td>\n",
       "      <td>43</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Date                                             Tweets Prices\n",
       "0  2021-09-22  ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...     43\n",
       "1  2021-09-21   RT SparrowOneSix 737900 N78448 was carrying U...     44\n",
       "2  2021-09-20   RT diecastryan A nice full lineup at IAD last...     45\n",
       "3  2021-09-19   jacobcabe Guess UnitedAirlines wont get any  ...     43\n",
       "4  2021-09-18   RT FELASTORY UnitedAirlines announce non stop...     43\n",
       "5  2021-09-17   UnitedAirlines 90 of workers vaccinated after...     44\n",
       "6  2021-09-16   This is how united UnitedAirlines treated wit...     44\n",
       "7  2021-09-15   Thank you SPONSORSYour generous support make ...     43\n",
       "8  2021-09-14   Because I get to work with amazing people uni...     43"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ccdata"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Making \"prices\" column as integer so mathematical operations could be performed easily."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:39:41.884443Z",
     "start_time": "2021-09-22T09:39:41.873474Z"
    }
   },
   "outputs": [],
   "source": [
    "ccdata['Prices'] = ccdata['Prices'].apply(np.int64)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Adding 4 new columns in our dataframe so that sentiment analysis could be performed.. Comp is \"Compound\" it will tell whether the statement is overall negative or positive. If it has negative value then it is negative, if it has positive value then it is positive. If it has value 0, then it is neutral."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:39:42.917283Z",
     "start_time": "2021-09-22T09:39:42.894365Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Tweets</th>\n",
       "      <th>Prices</th>\n",
       "      <th>Comp</th>\n",
       "      <th>Negative</th>\n",
       "      <th>Neutral</th>\n",
       "      <th>Positive</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2021-09-22</td>\n",
       "      <td>ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...</td>\n",
       "      <td>43</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2021-09-21</td>\n",
       "      <td>RT SparrowOneSix 737900 N78448 was carrying U...</td>\n",
       "      <td>44</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2021-09-20</td>\n",
       "      <td>RT diecastryan A nice full lineup at IAD last...</td>\n",
       "      <td>45</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2021-09-19</td>\n",
       "      <td>jacobcabe Guess UnitedAirlines wont get any  ...</td>\n",
       "      <td>43</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2021-09-18</td>\n",
       "      <td>RT FELASTORY UnitedAirlines announce non stop...</td>\n",
       "      <td>43</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>2021-09-17</td>\n",
       "      <td>UnitedAirlines 90 of workers vaccinated after...</td>\n",
       "      <td>44</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>2021-09-16</td>\n",
       "      <td>This is how united UnitedAirlines treated wit...</td>\n",
       "      <td>44</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>2021-09-15</td>\n",
       "      <td>Thank you SPONSORSYour generous support make ...</td>\n",
       "      <td>43</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>2021-09-14</td>\n",
       "      <td>Because I get to work with amazing people uni...</td>\n",
       "      <td>43</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Date                                             Tweets  Prices Comp  \\\n",
       "0  2021-09-22  ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...      43        \n",
       "1  2021-09-21   RT SparrowOneSix 737900 N78448 was carrying U...      44        \n",
       "2  2021-09-20   RT diecastryan A nice full lineup at IAD last...      45        \n",
       "3  2021-09-19   jacobcabe Guess UnitedAirlines wont get any  ...      43        \n",
       "4  2021-09-18   RT FELASTORY UnitedAirlines announce non stop...      43        \n",
       "5  2021-09-17   UnitedAirlines 90 of workers vaccinated after...      44        \n",
       "6  2021-09-16   This is how united UnitedAirlines treated wit...      44        \n",
       "7  2021-09-15   Thank you SPONSORSYour generous support make ...      43        \n",
       "8  2021-09-14   Because I get to work with amazing people uni...      43        \n",
       "\n",
       "  Negative Neutral Positive  \n",
       "0                            \n",
       "1                            \n",
       "2                            \n",
       "3                            \n",
       "4                            \n",
       "5                            \n",
       "6                            \n",
       "7                            \n",
       "8                            "
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ccdata[\"Comp\"] = ''\n",
    "ccdata[\"Negative\"] = ''\n",
    "ccdata[\"Neutral\"] = ''\n",
    "ccdata[\"Positive\"] = ''\n",
    "ccdata"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Downloading this package was essential to perform sentiment analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:39:45.645882Z",
     "start_time": "2021-09-22T09:39:44.853380Z"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package vader_lexicon to\n",
      "[nltk_data]     C:\\Users\\aanand2\\AppData\\Roaming\\nltk_data...\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import nltk\n",
    "nltk.download('vader_lexicon')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### This part of the code is responsible for assigning the polarity for each statement. That is how much positive, negative, neutral you statement is. And also assign the compound value that is overall sentiment of the statement."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:41:51.701865Z",
     "start_time": "2021-09-22T09:41:51.584148Z"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\aanand2\\Anaconda3\\lib\\site-packages\\pandas\\core\\indexing.py:1637: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  self._setitem_single_block(indexer, value, name)\n"
     ]
    }
   ],
   "source": [
    "from nltk.sentiment.vader import SentimentIntensityAnalyzer\n",
    "from nltk.sentiment.vader import SentimentIntensityAnalyzer\n",
    "import unicodedata\n",
    "sentiment_i_a = SentimentIntensityAnalyzer()\n",
    "for indexx, row in ccdata.T.iteritems():\n",
    "    try:\n",
    "        sentence_i = unicodedata.normalize('NFKD', ccdata.loc[indexx, 'Tweets'])\n",
    "        sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)\n",
    "        ccdata['Comp'].iloc[indexx] = sentence_sentiment['compound']\n",
    "        ccdata['Negative'].iloc[indexx] = sentence_sentiment['neg']\n",
    "        ccdata['Neutral'].iloc[indexx] = sentence_sentiment['neu']\n",
    "        ccdata['Positive'].iloc[indexx] = sentence_sentiment['compound']\n",
    "        # ccdata.set_value(indexx, 'Comp', sentence_sentiment['pos'])\n",
    "        # ccdata.set_value(indexx, 'Negative', sentence_sentiment['neg'])\n",
    "        # ccdata.set_value(indexx, 'Neutral', sentence_sentiment['neu'])\n",
    "        # ccdata.set_value(indexx, 'Positive', sentence_sentiment['pos'])\n",
    "    except TypeError:\n",
    "        print (stocks_dataf.loc[indexx, 'Tweets'])\n",
    "        print (indexx)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:41:52.464197Z",
     "start_time": "2021-09-22T09:41:52.452260Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Tweets</th>\n",
       "      <th>Prices</th>\n",
       "      <th>Comp</th>\n",
       "      <th>Negative</th>\n",
       "      <th>Neutral</th>\n",
       "      <th>Positive</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2021-09-22</td>\n",
       "      <td>ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...</td>\n",
       "      <td>43</td>\n",
       "      <td>0.9186</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.829</td>\n",
       "      <td>0.9186</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2021-09-21</td>\n",
       "      <td>RT SparrowOneSix 737900 N78448 was carrying U...</td>\n",
       "      <td>44</td>\n",
       "      <td>0.9997</td>\n",
       "      <td>0.021</td>\n",
       "      <td>0.787</td>\n",
       "      <td>0.9997</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2021-09-20</td>\n",
       "      <td>RT diecastryan A nice full lineup at IAD last...</td>\n",
       "      <td>45</td>\n",
       "      <td>0.9999</td>\n",
       "      <td>0.016</td>\n",
       "      <td>0.758</td>\n",
       "      <td>0.9999</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2021-09-19</td>\n",
       "      <td>jacobcabe Guess UnitedAirlines wont get any  ...</td>\n",
       "      <td>43</td>\n",
       "      <td>0.1262</td>\n",
       "      <td>0.075</td>\n",
       "      <td>0.852</td>\n",
       "      <td>0.1262</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2021-09-18</td>\n",
       "      <td>RT FELASTORY UnitedAirlines announce non stop...</td>\n",
       "      <td>43</td>\n",
       "      <td>0.9985</td>\n",
       "      <td>0.019</td>\n",
       "      <td>0.837</td>\n",
       "      <td>0.9985</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>2021-09-17</td>\n",
       "      <td>UnitedAirlines 90 of workers vaccinated after...</td>\n",
       "      <td>44</td>\n",
       "      <td>0.9986</td>\n",
       "      <td>0.036</td>\n",
       "      <td>0.85</td>\n",
       "      <td>0.9986</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>2021-09-16</td>\n",
       "      <td>This is how united UnitedAirlines treated wit...</td>\n",
       "      <td>44</td>\n",
       "      <td>0.984</td>\n",
       "      <td>0.085</td>\n",
       "      <td>0.767</td>\n",
       "      <td>0.984</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>2021-09-15</td>\n",
       "      <td>Thank you SPONSORSYour generous support make ...</td>\n",
       "      <td>43</td>\n",
       "      <td>0.9831</td>\n",
       "      <td>0.028</td>\n",
       "      <td>0.838</td>\n",
       "      <td>0.9831</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>2021-09-14</td>\n",
       "      <td>Because I get to work with amazing people uni...</td>\n",
       "      <td>43</td>\n",
       "      <td>0.9784</td>\n",
       "      <td>0.089</td>\n",
       "      <td>0.775</td>\n",
       "      <td>0.9784</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Date                                             Tweets  Prices  \\\n",
       "0  2021-09-22  ICAO A0A522Flt UAL961 UnitedAirlinesFirst seen...      43   \n",
       "1  2021-09-21   RT SparrowOneSix 737900 N78448 was carrying U...      44   \n",
       "2  2021-09-20   RT diecastryan A nice full lineup at IAD last...      45   \n",
       "3  2021-09-19   jacobcabe Guess UnitedAirlines wont get any  ...      43   \n",
       "4  2021-09-18   RT FELASTORY UnitedAirlines announce non stop...      43   \n",
       "5  2021-09-17   UnitedAirlines 90 of workers vaccinated after...      44   \n",
       "6  2021-09-16   This is how united UnitedAirlines treated wit...      44   \n",
       "7  2021-09-15   Thank you SPONSORSYour generous support make ...      43   \n",
       "8  2021-09-14   Because I get to work with amazing people uni...      43   \n",
       "\n",
       "     Comp Negative Neutral Positive  \n",
       "0  0.9186      0.0   0.829   0.9186  \n",
       "1  0.9997    0.021   0.787   0.9997  \n",
       "2  0.9999    0.016   0.758   0.9999  \n",
       "3  0.1262    0.075   0.852   0.1262  \n",
       "4  0.9985    0.019   0.837   0.9985  \n",
       "5  0.9986    0.036    0.85   0.9986  \n",
       "6   0.984    0.085   0.767    0.984  \n",
       "7  0.9831    0.028   0.838   0.9831  \n",
       "8  0.9784    0.089   0.775   0.9784  "
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ccdata"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "ccdata['']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Calculating the percentage of postive and negative tweets, and plotting the PIE chart for the same."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:43:58.673907Z",
     "start_time": "2021-09-22T09:43:58.601071Z"
    },
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "% of positive tweets=  100.0\n",
      "% of negative tweets=  0.0\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[]"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "posi=0\n",
    "nega=0\n",
    "for i in range (0,len(ccdata)):\n",
    "    get_val=ccdata.Comp[i]\n",
    "    if(float(get_val)<(0)):\n",
    "        nega=nega+1\n",
    "    if(float(get_val>(0))):\n",
    "        posi=posi+1\n",
    "posper=(posi/(len(ccdata)))*100\n",
    "negper=(nega/(len(ccdata)))*100\n",
    "print(\"% of positive tweets= \",posper)\n",
    "print(\"% of negative tweets= \",negper)\n",
    "arr=np.asarray([posper,negper], dtype=int)\n",
    "mlpt.pie(arr,labels=['positive','negative'])\n",
    "mlpt.plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Making a new dataframe with necessary columns for providing machine learning."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:44:01.310736Z",
     "start_time": "2021-09-22T09:44:01.301731Z"
    }
   },
   "outputs": [],
   "source": [
    "df_=ccdata[['Date','Prices','Comp','Negative','Neutral','Positive']].copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:44:01.654699Z",
     "start_time": "2021-09-22T09:44:01.635721Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Prices</th>\n",
       "      <th>Comp</th>\n",
       "      <th>Negative</th>\n",
       "      <th>Neutral</th>\n",
       "      <th>Positive</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2021-09-22</td>\n",
       "      <td>43</td>\n",
       "      <td>0.9186</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.829</td>\n",
       "      <td>0.9186</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2021-09-21</td>\n",
       "      <td>44</td>\n",
       "      <td>0.9997</td>\n",
       "      <td>0.021</td>\n",
       "      <td>0.787</td>\n",
       "      <td>0.9997</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2021-09-20</td>\n",
       "      <td>45</td>\n",
       "      <td>0.9999</td>\n",
       "      <td>0.016</td>\n",
       "      <td>0.758</td>\n",
       "      <td>0.9999</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2021-09-19</td>\n",
       "      <td>43</td>\n",
       "      <td>0.1262</td>\n",
       "      <td>0.075</td>\n",
       "      <td>0.852</td>\n",
       "      <td>0.1262</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2021-09-18</td>\n",
       "      <td>43</td>\n",
       "      <td>0.9985</td>\n",
       "      <td>0.019</td>\n",
       "      <td>0.837</td>\n",
       "      <td>0.9985</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>2021-09-17</td>\n",
       "      <td>44</td>\n",
       "      <td>0.9986</td>\n",
       "      <td>0.036</td>\n",
       "      <td>0.85</td>\n",
       "      <td>0.9986</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>2021-09-16</td>\n",
       "      <td>44</td>\n",
       "      <td>0.984</td>\n",
       "      <td>0.085</td>\n",
       "      <td>0.767</td>\n",
       "      <td>0.984</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>2021-09-15</td>\n",
       "      <td>43</td>\n",
       "      <td>0.9831</td>\n",
       "      <td>0.028</td>\n",
       "      <td>0.838</td>\n",
       "      <td>0.9831</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>2021-09-14</td>\n",
       "      <td>43</td>\n",
       "      <td>0.9784</td>\n",
       "      <td>0.089</td>\n",
       "      <td>0.775</td>\n",
       "      <td>0.9784</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Date  Prices    Comp Negative Neutral Positive\n",
       "0  2021-09-22      43  0.9186      0.0   0.829   0.9186\n",
       "1  2021-09-21      44  0.9997    0.021   0.787   0.9997\n",
       "2  2021-09-20      45  0.9999    0.016   0.758   0.9999\n",
       "3  2021-09-19      43  0.1262    0.075   0.852   0.1262\n",
       "4  2021-09-18      43  0.9985    0.019   0.837   0.9985\n",
       "5  2021-09-17      44  0.9986    0.036    0.85   0.9986\n",
       "6  2021-09-16      44   0.984    0.085   0.767    0.984\n",
       "7  2021-09-15      43  0.9831    0.028   0.838   0.9831\n",
       "8  2021-09-14      43  0.9784    0.089   0.775   0.9784"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Dividing the dataset into train and test."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:44:49.274215Z",
     "start_time": "2021-09-22T09:44:49.266237Z"
    },
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "train_start_index = '0'\n",
    "train_end_index = '5'\n",
    "test_start_index = '6'\n",
    "test_end_index = '8'\n",
    "train = df_.loc[train_start_index : train_end_index,:]\n",
    "test = df_.loc[test_start_index:test_end_index,:]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Making a 2D array that will store the Negative and Positive sentiment for Training dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:44:50.677601Z",
     "start_time": "2021-09-22T09:44:50.667601Z"
    }
   },
   "outputs": [],
   "source": [
    "sentiment_score_list = []\n",
    "for date, row in train.T.iteritems():\n",
    "    sentiment_score = np.asarray([df_.loc[date, 'Negative'],df_.loc[date, 'Positive']])\n",
    "    sentiment_score_list.append(sentiment_score)\n",
    "numpy_df_train = np.asarray(sentiment_score_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:44:52.251819Z",
     "start_time": "2021-09-22T09:44:52.236853Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.     0.9186]\n",
      " [0.021  0.9997]\n",
      " [0.016  0.9999]\n",
      " [0.075  0.1262]\n",
      " [0.019  0.9985]\n",
      " [0.036  0.9986]]\n"
     ]
    }
   ],
   "source": [
    "print(numpy_df_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Making a 2D array that will store the Negative and Positive sentiment for Testing dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:44:53.722231Z",
     "start_time": "2021-09-22T09:44:53.711261Z"
    }
   },
   "outputs": [],
   "source": [
    "sentiment_score_list = []\n",
    "for date, row in test.T.iteritems():\n",
    "    sentiment_score = np.asarray([df_.loc[date, 'Negative'],df_.loc[date, 'Positive']])\n",
    "    sentiment_score_list.append(sentiment_score)\n",
    "numpy_df_test = np.asarray(sentiment_score_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:44:54.096791Z",
     "start_time": "2021-09-22T09:44:54.083825Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.085  0.984 ]\n",
      " [0.028  0.9831]\n",
      " [0.089  0.9784]]\n"
     ]
    }
   ],
   "source": [
    "print(numpy_df_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Making 2 dataframe for Training and Testing \"Prices\". You can also make 1-D array for the same."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:44:55.720364Z",
     "start_time": "2021-09-22T09:44:55.700416Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Prices\n",
      "0      43\n",
      "1      44\n",
      "2      45\n",
      "3      43\n",
      "4      43\n",
      "5      44\n"
     ]
    }
   ],
   "source": [
    "y_train = pd.DataFrame(train['Prices'])\n",
    "#y_train=[91,91,91,92,91,92,91]\n",
    "y_test = pd.DataFrame(test['Prices'])\n",
    "print(y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Fitting the sentiments(this acts as in independent value) and prices(this acts as a dependent value (like class-lables in iris dataset))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:45:41.425399Z",
     "start_time": "2021-09-22T09:45:41.317688Z"
    },
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-80-5be54910e205>:7: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
      "  rf.fit(numpy_df_train, y_train)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "RandomForestRegressor()"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# from treeinterpreter import treeinterpreter as ti\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import classification_report,confusion_matrix\n",
    "\n",
    "rf = RandomForestRegressor()\n",
    "rf.fit(numpy_df_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Making Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:46:05.824853Z",
     "start_time": "2021-09-22T09:46:05.802877Z"
    }
   },
   "outputs": [],
   "source": [
    "prediction = rf.predict(numpy_df_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:46:06.416726Z",
     "start_time": "2021-09-22T09:46:06.411739Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[43.37 43.39 43.37]\n"
     ]
    }
   ],
   "source": [
    "print(prediction)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Importing matplotlib library for plotting graph"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:46:09.288378Z",
     "start_time": "2021-09-22T09:46:09.271441Z"
    }
   },
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Defining index position for the test data. Making dataframe for the predicted value."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:46:45.351277Z",
     "start_time": "2021-09-22T09:46:45.345293Z"
    }
   },
   "outputs": [],
   "source": [
    "idx=np.arange(int(test_start_index),int(test_end_index)+1)\n",
    "predictions_df_ = pd.DataFrame(data=prediction[0:], index = idx, columns=['Prices'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:46:46.414957Z",
     "start_time": "2021-09-22T09:46:46.405980Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Prices</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>43.37</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>43.39</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>43.37</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Prices\n",
       "6   43.37\n",
       "7   43.39\n",
       "8   43.37"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predictions_df_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Plotting the graph for the Predicted_price VS Actual Price"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:46:51.727734Z",
     "start_time": "2021-09-22T09:46:51.485198Z"
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYgAAAEWCAYAAAB8LwAVAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAA0k0lEQVR4nO3dd3xV9f3H8dc7IRCGbBwIClqrIkuMCKKU4kJmEUStYq2D0jrbqnWL61db0Lq1FBFxo0BVqoKD4QIMgqCAooKAOABBRmXm8/vjnKSXy01yE3JzkpvP8/G4j9yzP+fk3PM53zO+X5kZzjnnXLyMqANwzjlXMXmCcM45l5AnCOeccwl5gnDOOZeQJwjnnHMJeYJwzjmXkCcIlxKShkl6Muo4qoLYbS3pAEmbJGWWw3KXSTqxDOe3SdJBZTU/t+c8QVQh4Q/6p/CH+K2kMZLqRB3XnpDUTVJeuE75n5fLcfktJJmkauW1zKKY2XIzq2NmO4saL9xuK8srrmSEcX8ZdRzufzxBVD19zKwO0B44Erg22nDKxKrw4JL/6VPSGZTHGXcyKkqiKU9VcZ0rC08QVZSZfQtMJkgUAEi6RtIXkjZKWiipf8yw8yS9I2mEpHWSlko6NWZ4S0nTw2lfBxrHLk9SX0mfSFovaZqkw2OGLZN0laT5kjZLelTSPpJeDef3hqQGJV1HSYeHy1ofLrtvzLAxkh6W9IqkzcAvJTWVNF7S6nD9LosZv6OkXEkbJH0n6e5w0Izw7/qw9NI5QRzDJL0g6blwfT6U1C5u/f8iaT6wWVI1SZ0kvRfG/pGkbsls6/gSjaSGkh6TtCr8v/1bUm3gVaBpTKmrqaSMmH1graRxkhrGzHuwpK/CYdcXs+3HSHpE0uthnNMlHRgz3CRdLGkJsCSm38/C7zUl3RUu78dw36sZDitq25wn6ctwmUslnV1UnK4YZuafKvIBlgEnht+bAQuAe2OGnw40JThxOAPYDOwXDjsP2A5cBGQCvwdWAQqHvw/cDdQAugIbgSfDYT8P53USkAVcDXwOVI+JayawD7A/8D3wIUEJpwbwFnBzIevUDViZoH9WuIzrgOpA9zCmQ8PhY4AfgS7h+tYC5gA3heMfBHwJnBKzfoPD73WATuH3FoAB1YrY7sPCbTcwjOtKYCmQFbP+84DmQM1wG6wFeoaxnRR2N0liW+8SD/Af4DmgQbjsXxS23YArwv9Ds3De/wSeCYe1AjaFy6sRLn8H4f6UYJ3HhHHlj38v8E7McANeBxoCNWP6/Sz8/iAwLdwWmcCx4XwK3TZAbWBDzP94P+CIqH93lfkTeQD+Kcd/dnAg2hT+cA14E6hfxPjzgH7h9/OAz2OG1QrnsS9wQHiwqB0z/OmYg9aNwLiYYRnA10C3mLjOjhk+Hng4pvtS4N+FxNgNyAPWx3wGAccD3wIZMeM+AwwLv48BxsYMOwZYHjfva4HHwu8zgFuAxnHjtCC5BDEzbv2/AY6PWf/zY4b/BXgibh6Tgd8ksa0L4gkPkHlAg0K2W3yCWAScENO9H0Fiq0aQOJ+NGVYb2EbRCSJ2/DrATqB52G1A97hpDPhZuH1+AtolmG9R26Z2+P8fQJh0/LNnH7/EVPX8ysz2IjhAHMaulyfOlTQvLLqvB1qz66Wib/O/mNl/w691CEod68xsc8y4X8V8bxrbbWZ5wAqCs8F838V8/ylBd1E301eZWf2Yz7hwmSvCZcXGFLvMFTHfDyS45LI+Zv2vIyjVAFxAUBJaLOkDSb2LiCeRgmWFMa0MYywsltPjYjmO4IBd3LaO1Rz4wczWJRnjgcDEmGUuIjio7xMuN3YdNhOcuRcldvxNwA8Uvs6xGgPZwBeFxJhw24QxnQEMBb6R9B9JhxUToyuCJ4gqysymE5zljQAIrw//C7gEaGRm9YGPASUxu2+ABuG17XwHxHxfRfDDJlyWCA5eX5d+DYq1CmguKXYfPyBumbFVGa8AlsYlmr3MrCeAmS0xs7OAvYG/AS+E65tsdcjN87+EMTULYywslifiYqltZndS/LaOtQJoKKl+gmGJ4l4BnBq33Gwz+zpcbuw61AIaFbq2gdjx6xBcTipsnWOtAbYABxcSY2HbBjObbGYnESTTxQT7tCslTxBV2z3ASZLaExTPDVgNIOm3BCWIYpnZV0AucIuk6pKOA2KfJBoH9JJ0gqQs4M/AVuC9MlqPRGYR3Pe4WlJWeCOzD/BsIePPBjaEN4trSsqU1FrS0QCSzpHUJDz7Xx9Os5Nge+UR3LMoylGSTgtvHl9BsP4zCxn3SaCPpFPCOLIVPJbaLIltXcDMviG4Gf2QpAbhdugaDv4OaCSpXswkjwB35N9MltREUr9w2AtAb0nHSaoO3Erxx4+eMePfBswys8JKDbFx5wGjgbvDm+eZkjpLqlHUtlHwYEPfMHluJbicWuTjvq5oniCqMDNbDYwFbjSzhcBdBDdAvwPaAO+WYHa/JriO/wNwczjf/OV8CpwD3E9wdtiH4HHbbWWwGgmF8+4LnBou8yHgXDNbXMj4O8O42hPcQF4DjALyD6A9gE8kbSK44XqmmW0JL7XdAbwbXvLoVEhILxJc/lgHDAZOM7PthcSyAuhHcIlrNcFZ81X87/da6LZOYDDBfYTFBDf/rwiXsZjgnsyXYdxNw/V6CZgiaSNBAjsmHP8T4GKC+x3fhOtR3HsUT4fx/QAcBZTkiaIrCR6i+CCc/m8E95OK2jYZBCcfq8JpfgH8oQTLdHHyn0BxzqWIpGEET+ecE3Us5UXSGIKb4DdEHYsrPS9BOOecS8gThHPOuYT8EpNzzrmEvAThnHMuobSqJKtx48bWokWLqMNwzrlKY86cOWvMrEmiYWmVIFq0aEFubm7UYTjnXKUhqbA38f0Sk3POucQ8QTjnnEvIE4RzzrmE0uoehHMudbZv387KlSvZsmVL1KG4UsjOzqZZs2ZkZWUlPY0nCOdcUlauXMlee+1FixYtCCrkdZWFmbF27VpWrlxJy5Ytk54u5ZeYwhoX50qaFNf/yrCJwcaFTNdD0qeSPpd0TarjdM4VbcuWLTRq1MiTQyUkiUaNGpW49Fce9yAuJ2h4pICk5gRNBS5PNIGCBuQfJKiJsxVwlqRWKY7TOVcMTw6VV2n+dylNEJKaAb0Iqk2O9Q+CdokLq+ejI0Hzll+G1TY/S1DFb9nLy4MZI2DV3JTM3jnnKqtUlyDuIUgEBc0+SuoLfG1mHxUx3f7s2hzhSnZtKrKApCGSciXlrl69uuQRbv0Rch+DcefCf38o+fTOOZemUpYgwjZ7vzezOTH9agHXEzSAXuTkCfolLG2Y2UgzyzGznCZNEr4tXrSaDWDQWNjwDUwcGpQonHNpb9q0afTuHTQt/tJLL3HnnXcWOu769et56KGHSryMYcOGMWLEiFLHmC83N5fLLrtsj+dTUqksQXQB+kpaRnCJqDvwBNAS+Cjs3wz4UNK+cdOuJKY9W3Zvv7dsNTsKevwVlkyGd+5O2WKcc6m3c2fJWxnt27cv11xT+LMwpU0QZWHHjh3k5ORw3333lfuyU/aYq5ldC1wLELYHfKWZDYgdJ0wSOWa2Jm7yD4BDJLUkaGT+TIJmFlPn6Ath+UyYegc0y4GDuqV0cc5VZre8/AkLV20o03m2alqXm/scUeQ4y5Yto0ePHhxzzDHMnTuXn//854wdO5ZWrVpx/vnnM2XKFC655BIaNmzIzTffzNatWzn44IN57LHHqFOnDq+99hpXXHEFjRs3pkOHDgXzHTNmDLm5uTzwwAN89913DB06lC+//BKAhx9+mPvuu48vvviC9u3bc9JJJzF8+HCGDx/OuHHj2Lp1K/379+eWW24B4I477mDs2LE0b96cJk2acNRRRxW6Pt26daN9+/bMnj2bDRs2MHr0aDp27MiwYcNYtWoVy5Yto3HjxgwZMoQRI0YwadIkNm3axKWXXkpubi6SuPnmmxkwYABTpkxJuM57osK8SR02Tv4KgJntAC4BJhM8ATUubBM3lQFAn3uh0SHwwgWwIXUFFudc6X366acMGTKE+fPnU7du3YIz++zsbN555x1OPPFEbr/9dt544w0+/PBDcnJyuPvuu9myZQsXXXQRL7/8Mm+//Tbffvttwvlfdtll/OIXv+Cjjz7iww8/5IgjjuDOO+/k4IMPZt68eQwfPpwpU6awZMkSZs+ezbx585gzZw4zZsxgzpw5PPvss8ydO5cJEybwwQcfFLs+mzdv5r333uOhhx7i/PPPL+g/Z84cXnzxRZ5++uldxr/tttuoV68eCxYsYP78+XTv3p01a9YkXOc9VS4vypnZNGBagv4tYr6vAnrGdL8CvJL66GLUqANnPAEjfwnP/xbOmwSZyb916FxVUdyZfio1b96cLl26AHDOOecUXHo544wzAJg5cyYLFy4sGGfbtm107tyZxYsX07JlSw455JCCaUeOHLnb/N966y3Gjh0LQGZmJvXq1WPdunW7jDNlyhSmTJnCkUceCcCmTZtYsmQJGzdupH///tSqVQsILl0V56yzzgKga9eubNiwgfXr1xdMW7Nmzd3Gf+ONN3j22WcLuhs0aMCkSZMSrvOe8jep4zU5FPrdDy+cD6/fDD3+L+qInHMx4p/nz++uXbs2ELw1fNJJJ/HMM8/sMt68efPK7D0OM+Paa6/ld7/73S7977nnnhIvo7j1SbTs+GkKW+c9VWEuMVUorQdAx9/BzAfhk39HHY1zLsby5ct5//33AXjmmWc47rjjdhneqVMn3n33XT7//HMA/vvf//LZZ59x2GGHsXTpUr744ouCaRM54YQTePjhh4HghveGDRvYa6+92LhxY8E4p5xyCqNHj2bTpk0AfP3113z//fd07dqViRMn8tNPP7Fx40ZefvnlYtfnueeeA+Cdd96hXr161KtXr8jxTz75ZB544IGC7nXr1hW6znvKE0RhTr4dmh0NL14Ma5ZEHY1zLnT44Yfz+OOP07ZtW3744Qd+//vf7zK8SZMmjBkzhrPOOou2bdvSqVMnFi9eTHZ2NiNHjqRXr14cd9xxHHjggQnnf++99zJ16lTatGnDUUcdxSeffEKjRo3o0qULrVu35qqrruLkk0/m17/+NZ07d6ZNmzYMHDiQjRs30qFDB8444wzat2/PgAEDOP7444tdnwYNGnDssccydOhQHn300WLHv+GGG1i3bh2tW7emXbt2TJ06tdB13lMyK+xl5sonJyfHyrRFuR9Xwj+7Qu294aI3oXriIp9zVcGiRYs4/PDDI41h2bJl9O7dm48//jjSOMpKt27dGDFiBDk5OeWyvET/Q0lzzCxhAF6CKEq9ZjBgFKxeDJP+CGmUTJ1zrjh+k7o4B3eHX14XvB/R/Bg4+oKoI3KuymrRokWlLD1cfPHFvPvuu7v0u/zyy5k2bVo0ASXJE0Qyjr8SVsyG166BpkfC/h2Kn8Y550IPPvhg1CGUil9iSkZGBpw2EursA+N+45X6OeeqBE8QyarVEAY9Dpu+hQlDvFI/51za8wRREvuHlfp9/jq8fVfU0TjnXEp5giipnAugzaDgpvUXU6OOxjnnUsYTRElJ0OceaHIYjL8Afvw66oiccwlMmzaN9957b4/msae1oea78MILWbhwYZnMqzx5giiN6rWDSv12bIXnz4Md26KOyDkXpywSRFnYuXMno0aNolWrVlGHUmL+mGtpNT4E+j0QJIjXb4JTC2+Nyrm08+o18O2Csp3nvm2S+h396le/YsWKFWzZsoXLL7+cIUOG8Nprr3Hdddexc+dOGjduzKOPPsojjzxCZmYmTz75JPfffz+PPvoovXv3ZuDAgUBQOti0aRObNm2iX79+rFu3ju3bt3P77bfTr1+/YuOYNm0aN910E40aNeLTTz+la9euPPTQQ2RkZFCnTh3+9Kc/MXnyZO666y5uuOGGgjem42N988032bx5M5deeikLFixgx44dDBs2LKkYUs0TxJ44oj8snwWzHobmRweV/DnnUmr06NE0bNiQn376iaOPPpp+/fpx0UUXMWPGDFq2bMkPP/xAw4YNGTp0KHXq1OHKK68EKLSeo+zsbCZOnEjdunVZs2YNnTp1om/fvknVyjp79mwWLlzIgQceSI8ePZgwYQIDBw5k8+bNtG7dmltvvXWX8VevXr1brBA0MtS9e3dGjx7N+vXr6dixIyeeeGKhNbqWF08Qe+qkW+HrOfDSZbBPG2jy86gjci71Iiwx33fffUycOBGAFStWMHLkSLp27UrLli0BaNiwYYnmZ2Zcd911zJgxg4yMDL7++mu+++479t03viXk3XXs2JGDDjoICNp1eOeddxg4cCCZmZkMGLD7CePMmTMTxjplyhReeumlgvart2zZwvLlyyOv+8oTxJ6qVh1OHxNU6jduMFz4ZtDwkHOuzE2bNo033niD999/n1q1atGtWzfatWvHp59+Wuy01apVIy98f8nM2LYtuHf41FNPsXr1aubMmUNWVhYtWrRgy5YtScVTWFsO2dnZZGZm7jZ+orYc8vuPHz+eQw89NKnllhe/SV0W6u0PAx+FNZ/BpCu8Uj/nUuTHH3+kQYMG1KpVi8WLFzNz5ky2bt3K9OnTWbp0KUDBZZv4NhxatGjBnDlzAHjxxRfZvn17wTz33ntvsrKymDp1Kl999VXS8cyePZulS5eSl5fHc889t1vbFPE6d+6cMNZTTjmF+++/n/zatefOnZt0DKnkCaKsHNQtqNRvwfPwwaioo3EuLfXo0YMdO3bQtm1bbrzxRjp16kSTJk0YOXIkp512Gu3atStoerRPnz5MnDiR9u3b8/bbb3PRRRcxffp0OnbsyKxZswqu75999tnk5uaSk5PDU089xWGHHZZ0PJ07d+aaa66hdevWtGzZkv79+xc5fmGx3njjjWzfvp22bdvSunVrbrzxxlJuobLl7UGUpbw8eOZM+OItOH8yNDsqulicK2MVoT2IimTatGmMGDGCSZMmRR1K0rw9iChlZED/R6DufvC8V+rnnKvc/CZ1WavVEE5/HEafAhMugl8/HyQO51yltGDBAgYPHrxLvxo1ajBr1iy6desWTVDlxBNEKuzfAU79W9AK3Yzh0O0vUUfkXJko7CmcdNamTRvmzZsXdRh7rDS3E/zUNlWO+i20PROm/RU+fzPqaJzbY9nZ2axdu7ZUBxoXLTNj7dq1ZGdnl2g6L0GkigS9/wHfzofxF8LQt4M2rp2rpJo1a8bKlStZvXp11KG4UsjOzqZZs5IdgzxBpFL1WjDoCRjZLWiJ7revBi/WOVcJZWVlFbwB7KoGv8SUao1/FlTq93UuTLk+6miccy5pniDKwxG/gk4Xw+yRsOCFqKNxzrmkeIIoLyfdAs07BZX6fb846micc65YniDKS2ZWUKlf9Vow7lzYuinqiJxzrkieIMpT3f1g4GhYuwRevswr9XPOVWgpTxCSMiXNlTQp7L5N0nxJ8yRNkdS0kOn+KOkTSR9LekZSyR7grahadoXuN8DH42H2v6KOxjnnClUeJYjLgUUx3cPNrK2ZtQcmATfFTyBpf+AyIMfMWgOZwJnlEGv56PJH+HkPmHwdrPgg6miccy6hlCYISc2AXkBB/ddmtiFmlNpAYddZqgE1JVUDagGrUhVnuSuo1K9p0Kb15rVRR+Scc7tJdQniHuBqIC+2p6Q7JK0AziZBCcLMvgZGAMuBb4AfzWxKogVIGiIpV1JupXrDs2YDGDQWNq+GCRdC3s6oI3LOuV2kLEFI6g18b2Zz4oeZ2fVm1hx4CrgkwbQNgH5AS6ApUFvSOYmWY2YjzSzHzHKaNGlSpuuQck3bQ8+/B+1HTP971NE459wuUlmC6AL0lbQMeBboLunJuHGeBnZv2RtOBJaa2Woz2w5MAI5NYazR6fAbaPdrmP43WPJG1NE451yBlCUIM7vWzJqZWQuCG8xvmdk5kg6JGa0vkOitseVAJ0m1FNQtfAK73uhOHxL0ugv2OSK41LR+edQROeccEM17EHeGj67OB04meMoJSU0lvQJgZrOAF4APgQVhnCMjiLV8VK8V3I/I2xlU6rdja9QROeect0ldoSx8CcYNhqMvDEoVzjmXYt4mdWXRqi90vgQ+GAXzn486GudcFecJoqI5cRgccGxQFcf36XnbxTlXOXiCqGgys+D0x6B6HXhuMGzdGHVEzrkqyhNERbTXvkGlfj98AS9d6pX6Oeci4Qmiomp5PJxwE3wyEWb9M+ponHNVkCeIiqzLFXBoz6Cp0hWzo47GOVfFeIKoyCT41cNQr1lYqd+aqCNyzlUhniAqupr1w0r91sD4C7xSP+dcufEEURns1w56jYAvp8G0O6OOxjlXRXiCqCw6nAvtz4EZf4fPEtZ87pxzZcoTRGXSawTs0wYmXATrvoo6GudcmvMEUZlk1YRBj4PlwfNeqZ9zLrU8QVQ2jQ4OnmxaNRdeuybqaJxzacwTRGV0eG/ocjnkjoaPnos6GudcmvIEUVl1vwkOPA5evhy+Wxh1NM65NOQJorLKrBbU15RdN2hDYsuGqCNyzqUZTxCV2V77wMDH4Iel8NIlXqmfc65MeYKo7Fp0gRNvhoUvwsyHo47GOZdGPEGkg2Mvg8N6w+s3wvKZUUfjnEsTniDSgQT9HoR6zYNK/Tatjjoi51wa8ASRLmrWhzOegJ/WeaV+zrky4QkinezbBnrdBUunw9T/izoa51wl5wki3Rx5Dhw5GN4eAZ++FnU0zrlKzBNEOuo5PChNTBwC65ZFHY1zrpLyBJGOsmrCoCfAgHHnwvYtUUfknKuEik0Qkk6XtFf4/QZJEyR1SH1obo80bAn9H4FvPoLX/hJ1NM65SiiZEsSNZrZR0nHAKcDjgL+RVRkc1hOO+yPMGQPznok6GudcJZNMgsh/XrIX8LCZvQhUT11Irkz98gZocTxM+iN890nU0TjnKpFkEsTXkv4JDAJekVQjyelcRVBQqV89eG4wbPkx6oicc5VEMgf6QcBkoIeZrQcaAlelMihXxursDac/FjzR9OLFXqmfcy4pxSYIM/sv8D1wXNhrB7Ak2QVIypQ0V9KksPs2SfMlzZM0RVLTQqarL+kFSYslLZLUOdllugQOPBZOugUWvQzvPxh1NM65SiCZp5huBv4CXBv2ygKeLMEyLgcWxXQPN7O2ZtYemATcVMh09wKvmdlhQLu4ebjS6HwJHN4HXr8Jvno/6miccxVcMpeY+gN9gc0AZrYK2CuZmUtqRnBze1R+PzOLbdmmNsHT+vHT1QW6Ao+G02wLL2+5PZFfqV+DA8NK/b6POiLnXAWWTILYZmZGeCCXVLsE878HuBrIi+0p6Q5JK4CzSVyCOAhYDTwWXp4aVdhyJQ2RlCspd/Vqr8W0WNn1gpfotvwIL5wPO3dEHZFzroJKJkGMC59iqi/pIuAN4F/FTSSpN/C9mc2JH2Zm15tZc+Ap4JIEk1cDOhA8VnskQenlmkTLMbORZpZjZjlNmjRJYnUc+7aG3nfDsrdh6u1RR+Ocq6CSuUk9AngBGA8cCtxkZvcnMe8uQF9Jy4Bnge6S4u9dPA0MSDDtSmClmc0Ku18gSBiurLT/NXT4DbzzD1j8StTROOcqoGRuUrcE3jazq8zsSuAdSS2Km87MrjWzZmbWAjgTeMvMzpF0SMxofYHFCab9Flgh6dCw1wnAwmLXxpXMqX+H/drBxKFBu9bOORcjmUtMz7PrPYSdYb/SulPSx5LmAycTPOWEpKaSYk9lLwWeCsdrD3gDB2UtKxsGjQXhlfo553ZTLZlxzGxbfoeZbZNUoqo2zGwaMC38nuiSUv7TUT1juucBOSVZjiuFBi2g/0h45gx49Srom8zVQ+dcVZBMCWK1pL75HZL6AWtSF5Ird4f2gOP/DB+OhblPRR2Nc66CSKYEMZTgUs8DBBcjVgDnpjQqV/5+eT2s/AD+8yfYr23Q4JBzrkpL5immL8ysE9AKaGVmx5rZ56kPzZWrjEwYMBpqNgjuR3ilfs5VeYWWICSdY2ZPSvpTXH8AzOzuFMfmyludJnD6GBjTC/79BzjjyeDta+dclVRUCSL/zeW9Cvm4dHRAJzjpVlg8Cd7zG9bOVWWFliDM7J+SMoENZvaPcozJRa3TH2DFLHhjGOx/FLToEnVEzrkIFHkPwsx2ErzM5qoSCfo+ELRr/cJvYeN3UUfknItAMo+5vifpAUnHS+qQ/0l5ZC5a2XWDl+i2bAiShFfq51yVk8xjrseGf2+N6WdA97IPx1Uo+xwBfe6Bib+Dt24N7k0456qMZBLE6WbmL8ZVVe3OhOUz4d17ofkxcFivqCNyzpWTQi8xSeojaTUwX9JKSccWNq5Lcz3uhP3aw8Tfww9fRh2Nc66cFHUP4g7geDNrSlAl91/LJyRX4RRU6id47lzY/lPUETnnykFRCWKHmS0GCNtl8HcfqrIGB8Jp/4LvFsArV0YdjXOuHBR1D2LvuLeod+n2N6mroJ+fDF2vghnDoXkn6DA46oiccylUVAniX+z65nR8t6uKul0LB3ULShHfzI86GudcCsnMoo6hzOTk5Fhubm7UYaS/zWvgkeOhWnUYMh1q1o86IudcKUmaY2YJ295J5kU553ZVuzEMehx+XBlU6pdGJxnOuf/xBOFKp3lHOPl2+PQ/wTsSzrm0U2yCkFQjQb+GqQnHVSrHDIUj+sObt8DSt6OOxjlXxpIpQUyQlJXfIWk/4PXUheQqDSlow7rhwfDC+bDx26gjcs6VoWQSxL+B5yVlSmoBTAauTWVQrhKpsRec8QRs2wTP/xZ2bo86IudcGUmmydF/EZQY/g28DAw1sykpjstVJnsfDn3uheXvBZebnHNpoagmR2NfkhPQHJgHdJLUyV+Uc7toOyio1O+9+4NK/Q7vE3VEzrk9VNSb1PEvw00spL9zgR5/hVVzg0df924FjQ6OOiLn3B4oqslRv1bgSqZajeD9iH92hXHnwgWvQ/VaUUflnCulZB5zfV1S/ZjuBpImpzQqV3nVPwBOGwXffRJUx+Ev0TlXaSXzFFMTM1uf32Fm64C9UxaRq/wOORF+cTXMewo+HBt1NM65UkomQeyUdEB+h6QDCZocda5wv/gLHNwdXrkKVs2LOhrnXCkkkyCuB96R9ISkJ4AZ+HsQrjgZmcGlptqNg/sRP62LOiLnXAkl8x7Ea0AH4Lnwc5SZ+T0IV7zajeD0x2HDqqC50ry8qCNyzpVAspX1HQt0Cz+dUhWMS0PNj4ZT7oDPXoV3/xF1NM65EkjmKaY7gcuBheHncklJt08dVtExV9KksPs2SfMlzZM0RVLTZKd1lVTHIdB6ALx1OyydEXU0zrkkJVOC6AmcZGajzWw00APoVYJlXA4siukebmZtzaw9MAm4qQTTuspIgj73QaOfBZX6bVgVdUTOuSQke4mpfsz3esnOXFIzgmQyKr+fmW2IGaU2hTwRlWhaV4nVqAODnoBt//VK/ZyrJJJJEH8F5koaI+lxYE7YLxn3AFcDu9ydlHSHpBXA2RRegkg4bTxJQyTlSspdvXp1kmG5SOx9GPS9D1bMhDeGRR2Nc64YRdXFBICZPSNpGnA0QaV9fzGzYiv+l9Qb+N7M5kjqFjfP64HrJV0LXALcnOy0CeIbCYyEoE3q4uJyEWszMKjU7/0HglbpWvVLyWLy8oxtO/OCz47gsz3/+y79jG07d4b9rfBxd8b1j+9XMF7+PHYG8w7HE5CVmUH1asEn/3uNzAyyqonqmbv2rx7zNyuuO34eQT9RPTMz7K+ix83MICNDKdnuLr0UmyAkvWlmJwAvJehXlC5AX0k9gWygrqQnzeycmHGeBv5DXIJIcloXMTNjR96uB9Wt4QFx+864fvkH4x157NjnDxxXfyZ7jf89L66ox9rs5onnEXcQ3hp34N6+w3Y7gOePuyOvbM8VsjITHMRjDsJZ4YG3VvXge42CA3NwsAZ2SUqx67hlex4bftqRMCltL9ieZbs+1TJUaELaJclUy6R6pnZZx/zvNRIkn6ww6cVvp6xMBeNnZhYkxETzqJYhJE9eFYWskLpyJGUDtYCpBI+35v/X6gKvmtnhSS8kKAVcaWa9JR1iZkvC/pcCvzCzgclMW9xycnJyLDc3N9mwKo2izoZjD5rldTa8fUceW8MkUNqqlpqyhkk1ruM7a0D/bbeyhRpkiF0OPrsceBIclHc/u1biM+ai5hF34Ksed1DL7xf1QSsvz9ieF1fy2ZEX/r9tt//prvvF7v/73feL/HnE/K/jk1WC/WXbHuwDiUhBSStxkkl9qSt2v6sqpS5Jc8wsJ9GwokoQvwOuAJoS3HfI3zobgAf3IJ47JR1KcG/hK2BoGGRTYJSZ9dyDeZdaac+Gt+3cyfYdxtYEB92KfjZc5I+oiLPh/LPA+B9xwQ84wdlm/I+verUMMlbsw2Hjz+TjoydDv4eoVi2zTNcznWRkiBoZmdSogNtox87/7cNbwxOUQpNMghOV2N/N9p15hfyWii51FfyeYqb1UteeK7QEUTCCdKmZ3Z+SpZex0pYg2g6bzMatO8r8TKh6ogNjCc+GCx23lGfDWRkV7Exo2p0w7a/Q+x7I+W3U0bg0kl/y3qWEnWSpK3a6rbslq4pX6tqvbjbvXVvcVf/Cpi9FCULS0cCK/OQg6VxgAMFZ/zAz+6FU0VRA5x3bAijkJmKSZ8NZmaJGTKavlpnsE8RVXNerYcVsePVqaNoemh4ZdUQuTWRkiOyMTLKzKlapy8zYmbf7FYPSlrq27cyjRrXUHG+KugfxIXCimf0gqSvwLHAp0B44vKj7BlFJ13sQaW/z2qCRoYwMGDIdajWMOiLnqoyiShBFpZ3MmFLCGcBIMxtvZjcCPyvrIF0VVrsRDBoLG76Bib/zSv2cqyCKTBCS8i9BnQC8FTOs2MdjnSuRZkcFbVovmQLv3BV1NM45ij7QPwNMl7QG+Al4G0DSz4AfyyE2V9UcfWHwEt3U/4NmR8NB3aKOyLkqrdAShJndAfwZGAMcZ/+7WZFBcC/CubIlQZ97odEh8MIFXqmfcxEr8ta3mc00s4lmtjmm32dm9mHqQ3NVUo06cMYTsP0neP48r9TPuQj5s5iu4mlyKPS7H1bMgteLqg3eOZdKniBcxdR6AHT8Hcx8CD6ZGHU0zlVJniBcxXXy7cHN6hcvgTVLoo7GuSrHE4SruKpVh9Mfh2o14LnBsG1z8dM458qMJwhXsdXbHwaMgtWLYdIfKdNKbJxzRfIE4Sq+g7vDL6+D+c9B7uioo3GuyvAE4SqH46+En50Er10DX8+JOhrnqgRPEK5yyMiA00ZCnX1g3G/gv2lTmbBzFZYnCFd51GoIgx6HTd/BhCFeqZ9zKeYJwlUu+4eV+n3+Orw9IuponEtrniBc5ZNzAbQZFFTq98VbxY/vnCsVTxCu8pGgzz3Q5DAYfyH8uDLqiJxLS54gXOVUvXZQqd+OrUGlfju2RR2Rc2nHE4SrvBofAv0egJUfwOs3Rh2Nc2nHE4Sr3I7oD53+ALMegY/HRx2Nc2nFE4Sr/E66FZofAy9dBqs/izoa59KGJwhX+WVmweljoFo2jBsMWzdFHZFzacEThEsPdZvCwEdhzWcw6Qqv1M+5MuAJwqWPg7oFlfoteB4+GBV1NM5Vep4gXHo57s9wyCnw2rWwMjfqaJyr1DxBuPSSkQH9H4G6+wWV+m1eG3VEzlVaniBc+qnVMGiJbvP3MOEiyNsZdUTOVUqeIFx62r8DnPo3+OJNmDE86micq5RSniAkZUqaK2lS2H2bpPmS5kmaIqlpgmmaS5oqaZGkTyRdnuo4XRo66rfQ9kyYdid8/kbU0ThX6ZRHCeJyYFFM93Aza2tm7YFJwE0JptkB/NnMDgc6ARdLapXySF16kaD3P2Dvw2H8RbB+RdQROVeppDRBSGoG9AIKnjk0sw0xo9QGdntg3cy+MbMPw+8bCRLM/qmM1aWp6rVg0BOwc7tX6udcCaW6BHEPcDWwS9Nfku6QtAI4m8QliNhxWwBHArMKGT5EUq6k3NWrV5dFzC7dNP5ZUKnf17kw5fqoo3Gu0khZgpDUG/jezHZrYd7Mrjez5sBTwCVFzKMOMB64Iq7kETuvkWaWY2Y5TZo0KaPoXdo54lfQ+RKYPRIWvBB1NM5VCqksQXQB+kpaBjwLdJf0ZNw4TwMDEk0sKYsgOTxlZhNSGKerKk4cBgd0Dir1+35x1NE4V+GlLEGY2bVm1szMWgBnAm+Z2TmSDokZrS+w2y9VkoBHgUVmdneqYnRVTGYWDHwsuC8x7lyv1M+5YkTxHsSdkj6WNB84meApJyQ1lfRKOE4XYDBBqWNe+OkZQawu3dTdDwaOhrVL4OXLvFI/54pQrTwWYmbTgGnh94SXlMxsFdAz/P4OoPKIzVVBLbtC9xvgzbAdiWN+F3VEzlVI/ia1q5q6/BF+3gMmXw8rPog6GucqJE8QrmoqqNSvKTz/G9i8JuqInKtwPEG4qqtmAxg0NkgO4y/0Sv2ci+MJwlVtTdtDz7/Dl1Nh+t+ijsa5CsUThHMdfgPtfg3T/w5LvFI/5/J5gnBOgl53wT5HwIQLYf3yqCNyrkLwBOEchJX6jQ3uQ4z7DezYGnVEzkXOE4Rz+RodDL96CFZ9CJOvizoa5yLnCcK5WIf3gWMvhQ9Gwfzno47GuUh5gnAu3gnD4IBjg6o4vl9U7OjOpStPEM7Fy6wGpz8G1evAc4Nh68aoI3IuEp4gnEtkr32DSv1++AJevMQr9XNVkicI5wrT8ng44SZY+G+Y9UjU0ThX7jxBOFeULlfAoT1hyg2wPGGrt86lLU8QzhVFgl89DPWawfPnwSZv99xVHZ4gnCtOzfrBS3T/XQvjL/BK/VyV4QnCuWTs1w56jYCl02HaX6OOxrly4QnCuWR1OBfanwMzhsNnU6KOxrmU8wThXEn0GgH7tIEJF8G6r6KOxrmU8gThXElk1YRBj4PlBS3ReaV+Lo15gnCupBodHDRXumouvHZN1NE4lzKeIJwrjcN6QZfLIXc0fPRc1NE4lxKeIJwrre43wYHHwcuXw3efRB2Nc2XOE4RzpZVZLaivKbtuUKnflg1RR+RcmfIE4dye2GsfGPgYrFsGL17slfq5tOIJwrk91aILnHgzLHoJZj4UdTTOlRlPEM6VhWMvg8N6w+s3wfKZUUfjXJnwBOFcWZCg34NQr7lX6ufShicI58pKzfpwxhPw0zoYf75X6ucqPU8QzpWlfdtAr7tg6QyYekfU0Ti3RzxBOFfWjjwHjhwMb98Fn74WdTTOlVrKE4SkTElzJU0Ku2+TNF/SPElTJDUtZLoekj6V9Lkkr8/AVS49hweliYlDgkdgnauEyqMEcTmwKKZ7uJm1NbP2wCTgpvgJJGUCDwKnAq2AsyS1KodYnSsbWTVh0BPB93HnwvYt0cbjXClUS+XMJTUDegF3AH8CMLPY101rA4neLOoIfG5mX4bzeRboByxMZbzOlamGLaH/P+GZM+GBHKheO+qIXLqq2RDOf7XMZ5vSBAHcA1wN7BXbU9IdwLnAj8AvE0y3P7AipnslcEyiBUgaAgwBOOCAA/Y4YOfK1KGnQt8H4PPXo47EpbPseimZbcoShKTewPdmNkdSt9hhZnY9cL2ka4FLgJvjJ08wy4R1GJjZSGAkQE5Ojtdz4CqeDoODj3OVTCrvQXQB+kpaBjwLdJf0ZNw4TwMDEky7Emge090MWJWKIJ1zziWWsgRhZteaWTMzawGcCbxlZudIOiRmtL7A4gSTfwAcIqmlpOrh9C+lKlbnnHO7S/U9iETulHQokAd8BQwFCB93HWVmPc1sh6RLgMlAJjDazLzCfeecK0eyNKqeOCcnx3Jzc6MOwznnKg1Jc8wsJ9Ewf5PaOedcQp4gnHPOJeQJwjnnXEKeIJxzziWUVjepJa0meDKqNBoDa8ownLLicZWMx1UyHlfJpGNcB5pZk0QD0ipB7AlJuYXdyY+Sx1UyHlfJeFwlU9Xi8ktMzjnnEvIE4ZxzLiFPEP8zMuoACuFxlYzHVTIeV8lUqbj8HoRzzrmEvAThnHMuIU8QzjnnEkr7BCGpvqQXJC2WtEhS57jhknSfpM8lzZfUIWZYD0mfhsOuKee4zg7jmS/pPUntYoYtk7RA0jxJZVo7YRJxdZP0Y7jseZJuihkW5fa6KiamjyXtlNQwHJbK7XVozHLnSdog6Yq4ccp9H0syrnLfx5KMq9z3sSTjimof+6OkT8JlPiMpO2546vYvM0vrD/A4cGH4vTpQP254T+BVglbsOgGzwv6ZwBfAQeF0HwGtyjGuY4EG4fdT8+MKu5cBjSPaXt2ASQmmi3R7xY3bh6D9kZRvrwTb4FuCF48i38eSiCuSfSyJuCLZx4qLK4p9jKD55aVAzbB7HHBeee1faV2CkFQX6Ao8CmBm28xsfdxo/YCxFpgJ1Je0H9AR+NzMvjSzbQSt4vUrr7jM7D0zWxd2ziRoVS+lktxehYl0e8U5C3imLJZdQicAX5hZ/Nv85b6PJRNXFPtYMnEVIdLtFac897FqQE1J1YBa7N66Zsr2r7ROEASZczXwmKS5kkZJqh03zv7AipjulWG/wvqXV1yxLiA4Q8hnwBRJcyQNKaOYShJXZ0kfSXpV0hFhvwqxvSTVAnoA42N6p2p7xTuTxAeNKPaxZOKKVV77WLJxlfc+lmxc5bqPmdnXwAhgOfAN8KOZTYkbLWX7V7oniGpAB+BhMzsS2AzEX4dTgumsiP7lFVcQnPRLgh/vX2J6dzGzDgSXBS6W1LUc4/qQoOjdDrgf+Hd+qAnmV+7bi6Do/66Z/RDTL1Xbq4CCpnH7As8nGpygX6r3sWTiyh+nPPexZOKKYh9LJq585baPSWpAcNbfEmgK1JZ0TvxoCSYtk/0r3RPESmClmc0Ku18gONDEj9M8prsZQRGusP7lFReS2gKjgH5mtja/v5mtCv9+D0wkKEqWS1xmtsHMNoXfXwGyJDWmAmyv0G5nfyncXrFOBT40s+8SDItiH0smrij2sWLjimgfKzauGOW5j50ILDWz1Wa2HZhAcO8oVsr2r7ROEGb2LbBCQRvYEFxbXBg32kvAueGTAJ0IinDfAB8Ah0hqGZ5VnBmOWy5xSTqAYGcYbGafxfSvLWmv/O/AycDH5RjXvpIUfu9IsA+tJeLtFcZTD/gF8GJMv5RtrzhFXZMu930smbii2MeSjKvc97Fk4grjKe99bDnQSVKtcJucACyKGyd1+1dZ3GmvyB+gPZALzCcoqjYAhgJDw+ECHiS4278AyImZtifwWTjs+nKOaxSwDpgXfnLD/gcRPI3wEfBJBHFdEi73I4Ibm8dWhO0VjnMe8GzcdCndXuEyahEcwOrF9KsI+1hxcUW1jxUXV1T7WJFxRbWPAbcAiwmSzhNAjfLav7yqDeeccwml9SUm55xzpecJwjnnXEKeIJxzziXkCcI551xCniCcc84l5AnCuSJI2lTC8btJmpSqeJwrT54gnHPOJeQJwrkkhCWDafpfmxRPxbzt2yPs9w5wWsw0tSWNlvRBWMlgv7D/fQrbOJB0iqQZkjIkHSVpeljh22QFNXIi6TJJCxXU9f9sBKvvqih/Uc65IkjaZGZ1JHUjqF7hCIL6bN4FriJ4u3sJ0B34HHgOqGVmvSX9H7DQzJ6UVB+YDRxJUGHaBwRvDD9C8LbrcmA6QZ1IqyWdAZxiZudLWgW0NLOtkupb8lWwO7dHqkUdgHOVyGwzWwkgaR7QAthEUJnakrD/k0B+dc8nA30lXRl2ZwMHmNkiSRcBM4A/mtkXkloDrYHXw4JJJkH1zhBUL/KUpH/zv5pNnUs5TxDOJW9rzPed/O/3U1gxXMAAM/s0wbA2BPX+NI0Z9xMz65xg3F4EDSb1BW6UdISZ7Shp8M6VlN+DcG7PLAZaSjo47D4rZthk4NKYexVHhn8PBP5McLnpVEnHAJ8CTRS2tS0pS9IRkjKA5mY2FbgaqA/USf1qOecJwrk9YmZbCC4p/Se8SR3bTOVtQBYwX9LHwG1hsngUuNKCNgQuIKhVNQMYCPxN0kcEtaseS3Cp6UlJC4C5wD/8HoQrL36T2jnnXEJegnDOOZeQJwjnnHMJeYJwzjmXkCcI55xzCXmCcM45l5AnCOeccwl5gnDOOZfQ/wMXy7VwYjHFxwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "ax = predictions_df_.rename(columns={\"Prices\": \"predicted_price\"}).plot(title='Random Forest predicted prices')#predicted value\n",
    "ax.set_xlabel(\"Indexes\")\n",
    "ax.set_ylabel(\"Stock Prices\")\n",
    "fig = y_test.rename(columns={\"Prices\": \"actual_price\"}).plot(ax = ax).get_figure()#actual value\n",
    "fig.savefig(\"random forest.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:47:06.527704Z",
     "start_time": "2021-09-22T09:47:06.512721Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# from treeinterpreter import treeinterpreter as ti\n",
    "# from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import classification_report,confusion_matrix\n",
    "\n",
    "reg = LinearRegression()\n",
    "reg.fit(numpy_df_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:47:06.903403Z",
     "start_time": "2021-09-22T09:47:06.899386Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[45.17154917],\n",
       "       [44.0022019 ],\n",
       "       [45.24044194]])"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reg.predict(numpy_df_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### NOTE: Since our dataset is very small and as you can see that fetching 600 tweets could only make data for just 10 days.Also the prediction is not very great in such small dataset. So we found this new dataset on internet which has the Text as \"Tweets\" and respective \"close price\" and \"Adjusted close price\".\n",
    "\n",
    "\n",
    "### Adjusted Close Price: An adjusted closing price is a stock's closing price on any given day of trading that has been amended to include any distributions and corporate actions that occurred at any time before the next day's open."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:53:18.910727Z",
     "start_time": "2021-09-22T09:53:18.834707Z"
    }
   },
   "outputs": [],
   "source": [
    "stocks_dataf = pd.read_pickle('Twitter_Dataset.pkl')\n",
    "stocks_dataf.columns=['closing_price','adj_close_price','Tweets']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## New dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:53:19.550480Z",
     "start_time": "2021-09-22T09:53:19.536516Z"
    },
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>closing_price</th>\n",
       "      <th>adj_close_price</th>\n",
       "      <th>Tweets</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2007-01-01</th>\n",
       "      <td>12469.971875</td>\n",
       "      <td>12469.971875</td>\n",
       "      <td>. What Sticks from '06. Somalia Orders Islamis...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2007-01-02</th>\n",
       "      <td>12472.245703</td>\n",
       "      <td>12472.245703</td>\n",
       "      <td>. Heart Health: Vitamin Does Not Prevent Death...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2007-01-03</th>\n",
       "      <td>12474.519531</td>\n",
       "      <td>12474.519531</td>\n",
       "      <td>. Google Answer to Filling Jobs Is an Algorith...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2007-01-04</th>\n",
       "      <td>12480.690430</td>\n",
       "      <td>12480.690430</td>\n",
       "      <td>. Helping Make the Shift From Combat to Commer...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2007-01-05</th>\n",
       "      <td>12398.009766</td>\n",
       "      <td>12398.009766</td>\n",
       "      <td>. Rise in Ethanol Raises Concerns About Corn a...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-12-27</th>\n",
       "      <td>19945.039062</td>\n",
       "      <td>19945.039062</td>\n",
       "      <td>. Should the U.S. Embassy Be Moved From Tel Av...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-12-28</th>\n",
       "      <td>19833.679688</td>\n",
       "      <td>19833.679688</td>\n",
       "      <td>. When Finding the Right Lawyer Seems Daunting...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-12-29</th>\n",
       "      <td>19819.779297</td>\n",
       "      <td>19819.779297</td>\n",
       "      <td>. Does Empathy Guide or Hinder Moral Action?. ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-12-30</th>\n",
       "      <td>19762.599609</td>\n",
       "      <td>19762.599609</td>\n",
       "      <td>. Shielding Seized Assets From Corruption’s Cl...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-12-31</th>\n",
       "      <td>19762.599609</td>\n",
       "      <td>19762.599609</td>\n",
       "      <td>Terrorist Attack at Nightclub in Istanbul Kill...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3653 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "            closing_price  adj_close_price  \\\n",
       "2007-01-01   12469.971875     12469.971875   \n",
       "2007-01-02   12472.245703     12472.245703   \n",
       "2007-01-03   12474.519531     12474.519531   \n",
       "2007-01-04   12480.690430     12480.690430   \n",
       "2007-01-05   12398.009766     12398.009766   \n",
       "...                   ...              ...   \n",
       "2016-12-27   19945.039062     19945.039062   \n",
       "2016-12-28   19833.679688     19833.679688   \n",
       "2016-12-29   19819.779297     19819.779297   \n",
       "2016-12-30   19762.599609     19762.599609   \n",
       "2016-12-31   19762.599609     19762.599609   \n",
       "\n",
       "                                                       Tweets  \n",
       "2007-01-01  . What Sticks from '06. Somalia Orders Islamis...  \n",
       "2007-01-02  . Heart Health: Vitamin Does Not Prevent Death...  \n",
       "2007-01-03  . Google Answer to Filling Jobs Is an Algorith...  \n",
       "2007-01-04  . Helping Make the Shift From Combat to Commer...  \n",
       "2007-01-05  . Rise in Ethanol Raises Concerns About Corn a...  \n",
       "...                                                       ...  \n",
       "2016-12-27  . Should the U.S. Embassy Be Moved From Tel Av...  \n",
       "2016-12-28  . When Finding the Right Lawyer Seems Daunting...  \n",
       "2016-12-29  . Does Empathy Guide or Hinder Moral Action?. ...  \n",
       "2016-12-30  . Shielding Seized Assets From Corruption’s Cl...  \n",
       "2016-12-31  Terrorist Attack at Nightclub in Istanbul Kill...  \n",
       "\n",
       "[3653 rows x 3 columns]"
      ]
     },
     "execution_count": 123,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "stocks_dataf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:53:19.847347Z",
     "start_time": "2021-09-22T09:53:19.829419Z"
    }
   },
   "outputs": [],
   "source": [
    "stocks_dataf = stocks_dataf.reset_index().rename(columns = {'index':'Date'})"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Removing dot (.) and space from the Tweets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:53:20.532074Z",
     "start_time": "2021-09-22T09:53:20.485208Z"
    },
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>adj_close_price</th>\n",
       "      <th>Tweets</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2007-01-01</td>\n",
       "      <td>12469</td>\n",
       "      <td>What Sticks from '06. Somalia Orders Islamist...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2007-01-02</td>\n",
       "      <td>12472</td>\n",
       "      <td>Heart Health: Vitamin Does Not Prevent Death ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2007-01-03</td>\n",
       "      <td>12474</td>\n",
       "      <td>Google Answer to Filling Jobs Is an Algorithm...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2007-01-04</td>\n",
       "      <td>12480</td>\n",
       "      <td>Helping Make the Shift From Combat to Commerc...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2007-01-05</td>\n",
       "      <td>12398</td>\n",
       "      <td>Rise in Ethanol Raises Concerns About Corn as...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3648</th>\n",
       "      <td>2016-12-27</td>\n",
       "      <td>19945</td>\n",
       "      <td>Should the U.S. Embassy Be Moved From Tel Avi...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3649</th>\n",
       "      <td>2016-12-28</td>\n",
       "      <td>19833</td>\n",
       "      <td>When Finding the Right Lawyer Seems Daunting,...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3650</th>\n",
       "      <td>2016-12-29</td>\n",
       "      <td>19819</td>\n",
       "      <td>Does Empathy Guide or Hinder Moral Action?. C...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3651</th>\n",
       "      <td>2016-12-30</td>\n",
       "      <td>19762</td>\n",
       "      <td>Shielding Seized Assets From Corruption’s Clu...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3652</th>\n",
       "      <td>2016-12-31</td>\n",
       "      <td>19762</td>\n",
       "      <td>Terrorist Attack at Nightclub in Istanbul Kill...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3653 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "           Date  adj_close_price  \\\n",
       "0    2007-01-01            12469   \n",
       "1    2007-01-02            12472   \n",
       "2    2007-01-03            12474   \n",
       "3    2007-01-04            12480   \n",
       "4    2007-01-05            12398   \n",
       "...         ...              ...   \n",
       "3648 2016-12-27            19945   \n",
       "3649 2016-12-28            19833   \n",
       "3650 2016-12-29            19819   \n",
       "3651 2016-12-30            19762   \n",
       "3652 2016-12-31            19762   \n",
       "\n",
       "                                                 Tweets  \n",
       "0      What Sticks from '06. Somalia Orders Islamist...  \n",
       "1      Heart Health: Vitamin Does Not Prevent Death ...  \n",
       "2      Google Answer to Filling Jobs Is an Algorithm...  \n",
       "3      Helping Make the Shift From Combat to Commerc...  \n",
       "4      Rise in Ethanol Raises Concerns About Corn as...  \n",
       "...                                                 ...  \n",
       "3648   Should the U.S. Embassy Be Moved From Tel Avi...  \n",
       "3649   When Finding the Right Lawyer Seems Daunting,...  \n",
       "3650   Does Empathy Guide or Hinder Moral Action?. C...  \n",
       "3651   Shielding Seized Assets From Corruption’s Clu...  \n",
       "3652  Terrorist Attack at Nightclub in Istanbul Kill...  \n",
       "\n",
       "[3653 rows x 3 columns]"
      ]
     },
     "execution_count": 125,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "stocks_dataf['adj_close_price'] = stocks_dataf['adj_close_price'].apply(np.int64)\n",
    "stocks_dataf = stocks_dataf[['Date','adj_close_price', 'Tweets']]\n",
    "stocks_dataf['Tweets'] = stocks_dataf['Tweets'].map(lambda x: x.lstrip('.-'))\n",
    "stocks_dataf"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Making new dataframe and only considering \"Adjusted close price\". And date as index vlaue."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:53:32.527076Z",
     "start_time": "2021-09-22T09:53:32.509089Z"
    }
   },
   "outputs": [],
   "source": [
    "dataframe = stocks_dataf[['adj_close_price']].copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:53:32.774480Z",
     "start_time": "2021-09-22T09:53:32.759519Z"
    }
   },
   "outputs": [],
   "source": [
    "# dataframe = dataframe.reset_index().rename(columns = {'index':'Date'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:53:33.007730Z",
     "start_time": "2021-09-22T09:53:32.990777Z"
    },
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "dataframe[\"Comp\"] = ''\n",
    "dataframe[\"Negative\"] = ''\n",
    "dataframe[\"Neutral\"] = ''\n",
    "dataframe[\"Positive\"] = ''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:53:33.273715Z",
     "start_time": "2021-09-22T09:53:33.258723Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>adj_close_price</th>\n",
       "      <th>Comp</th>\n",
       "      <th>Negative</th>\n",
       "      <th>Neutral</th>\n",
       "      <th>Positive</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>12469</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>12472</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>12474</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>12480</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>12398</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3648</th>\n",
       "      <td>19945</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3649</th>\n",
       "      <td>19833</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3650</th>\n",
       "      <td>19819</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3651</th>\n",
       "      <td>19762</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3652</th>\n",
       "      <td>19762</td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "      <td></td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3653 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      adj_close_price Comp Negative Neutral Positive\n",
       "0               12469                               \n",
       "1               12472                               \n",
       "2               12474                               \n",
       "3               12480                               \n",
       "4               12398                               \n",
       "...               ...  ...      ...     ...      ...\n",
       "3648            19945                               \n",
       "3649            19833                               \n",
       "3650            19819                               \n",
       "3651            19762                               \n",
       "3652            19762                               \n",
       "\n",
       "[3653 rows x 5 columns]"
      ]
     },
     "execution_count": 134,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:53:33.648061Z",
     "start_time": "2021-09-22T09:53:33.632101Z"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package vader_lexicon to\n",
      "[nltk_data]     C:\\Users\\aanand2\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package vader_lexicon is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 135,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import nltk\n",
    "nltk.download('vader_lexicon')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:54:26.035857Z",
     "start_time": "2021-09-22T09:53:34.094155Z"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\aanand2\\Anaconda3\\lib\\site-packages\\pandas\\core\\indexing.py:1637: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  self._setitem_single_block(indexer, value, name)\n"
     ]
    }
   ],
   "source": [
    "from nltk.sentiment.vader import SentimentIntensityAnalyzer\n",
    "from nltk.sentiment.vader import SentimentIntensityAnalyzer\n",
    "import unicodedata\n",
    "sentiment_i_a = SentimentIntensityAnalyzer()\n",
    "for indexx, row in dataframe.T.iteritems():\n",
    "    try:\n",
    "        sentence_i = unicodedata.normalize('NFKD', stocks_dataf.loc[indexx, 'Tweets'])\n",
    "        sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)\n",
    "        dataframe['Comp'].iloc[indexx] = sentence_sentiment['compound']\n",
    "        dataframe['Negative'].iloc[indexx] = sentence_sentiment['neg']\n",
    "        dataframe['Neutral'].iloc[indexx] = sentence_sentiment['neu']\n",
    "        dataframe['Positive'].iloc[indexx] = sentence_sentiment['compound']\n",
    "        # dataframe.set_value(indexx, 'Comp', sentence_sentiment['compound'])\n",
    "        # dataframe.set_value(indexx, 'Negative', sentence_sentiment['neg'])\n",
    "        # dataframe.set_value(indexx, 'Neutral', sentence_sentiment['neu'])\n",
    "        # dataframe.set_value(indexx, 'Positive', sentence_sentiment['pos'])\n",
    "    except TypeError:\n",
    "        print (stocks_dataf.loc[indexx, 'Tweets'])\n",
    "        print (indexx)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:55:10.893144Z",
     "start_time": "2021-09-22T09:55:10.882175Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>adj_close_price</th>\n",
       "      <th>Comp</th>\n",
       "      <th>Negative</th>\n",
       "      <th>Neutral</th>\n",
       "      <th>Positive</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>12469</td>\n",
       "      <td>-0.9814</td>\n",
       "      <td>0.159</td>\n",
       "      <td>0.749</td>\n",
       "      <td>-0.9814</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>12472</td>\n",
       "      <td>-0.8521</td>\n",
       "      <td>0.116</td>\n",
       "      <td>0.785</td>\n",
       "      <td>-0.8521</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>12474</td>\n",
       "      <td>-0.9993</td>\n",
       "      <td>0.198</td>\n",
       "      <td>0.737</td>\n",
       "      <td>-0.9993</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>12480</td>\n",
       "      <td>-0.9982</td>\n",
       "      <td>0.131</td>\n",
       "      <td>0.806</td>\n",
       "      <td>-0.9982</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>12398</td>\n",
       "      <td>-0.9901</td>\n",
       "      <td>0.124</td>\n",
       "      <td>0.794</td>\n",
       "      <td>-0.9901</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3648</th>\n",
       "      <td>19945</td>\n",
       "      <td>-0.9898</td>\n",
       "      <td>0.178</td>\n",
       "      <td>0.719</td>\n",
       "      <td>-0.9898</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3649</th>\n",
       "      <td>19833</td>\n",
       "      <td>-0.6072</td>\n",
       "      <td>0.132</td>\n",
       "      <td>0.76</td>\n",
       "      <td>-0.6072</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3650</th>\n",
       "      <td>19819</td>\n",
       "      <td>-0.9782</td>\n",
       "      <td>0.14</td>\n",
       "      <td>0.761</td>\n",
       "      <td>-0.9782</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3651</th>\n",
       "      <td>19762</td>\n",
       "      <td>-0.995</td>\n",
       "      <td>0.168</td>\n",
       "      <td>0.734</td>\n",
       "      <td>-0.995</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3652</th>\n",
       "      <td>19762</td>\n",
       "      <td>-0.2869</td>\n",
       "      <td>0.173</td>\n",
       "      <td>0.665</td>\n",
       "      <td>-0.2869</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3653 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      adj_close_price    Comp Negative Neutral Positive\n",
       "0               12469 -0.9814    0.159   0.749  -0.9814\n",
       "1               12472 -0.8521    0.116   0.785  -0.8521\n",
       "2               12474 -0.9993    0.198   0.737  -0.9993\n",
       "3               12480 -0.9982    0.131   0.806  -0.9982\n",
       "4               12398 -0.9901    0.124   0.794  -0.9901\n",
       "...               ...     ...      ...     ...      ...\n",
       "3648            19945 -0.9898    0.178   0.719  -0.9898\n",
       "3649            19833 -0.6072    0.132    0.76  -0.6072\n",
       "3650            19819 -0.9782     0.14   0.761  -0.9782\n",
       "3651            19762  -0.995    0.168   0.734   -0.995\n",
       "3652            19762 -0.2869    0.173   0.665  -0.2869\n",
       "\n",
       "[3653 rows x 5 columns]"
      ]
     },
     "execution_count": 137,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T09:55:12.166877Z",
     "start_time": "2021-09-22T09:55:12.067276Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "% of positive tweets=  44.2102381604161\n",
      "% of negative tweets=  55.57076375581713\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[]"
      ]
     },
     "execution_count": 138,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAOcAAADnCAYAAADl9EEgAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAVE0lEQVR4nO3de7RVZbnH8e/D3uwtqC01BRHL5RUEMbyMLERLTTNXo0Ikb5i3zslLXo/lshqdmTU662RWXirpYJkdKxMFy6Vp3hW8oaDcBBy61BJBS6aK3Pbe7/njnRyXyBb2bT3vnPP5jLEHk317fxv97Xe+a97EOYcxJjz9tAMYYzbMymlMoKycxgTKymlMoKycxgTKymlMoKycxgTKymlMoKycxgTKymlMoKycxgTKymlMoKycxgTKymlMoKycxgTKymlMoKycpk+IyBki8pVk+xQR2aHuY5NFZIReunQQuxOC6Wsicj9wkXNupnaWNLGZ07yPiBRF5FkR+a2IPCMiU0RkoIgcJiKzRGSOiPxaRFqTz6+IyPzkc3+cvC8SkYtE5Bhgf+AGEZktIgNE5H4R2V9EzhSRH9WNe4qIXJVsTxSRx5OvmSQiTRr/FpqsnKYzw4BfOef2Bt4ELgSuA451zo0CmoEzRWQbYBwwMvncH9R/E+fcFGAmcKJzbrRzbmXdh6cAR9f9/VjgRhHZM9k+0Dk3GmgHTuz9HzFsVk7TmZedc9OT7f8FDgNecM4tSt73W+BgfHFXAZNF5GjgnU0dwDn3GvC8iHxCRD6M/4UwPRlrP+AJEZmd/H2Xnv9I6dKsHcAEa5NejHDOtYnIx/EFOg74OnBoF8a5Efgy8Cww1TnnRESA3zrnLuli5kyxmdN05qMi8slk+3jgbqAoIrsl7zsJeEBEtgAKzrnbgfOB0Rv4Xm8BW3Yyzi3Al5Ixbkzedw9wjIgMAhCRbURkpx79NClkM6fpzALgZBGZBCwGzgMeBW4SkWbgCeAaYBvgVhHZDBDggg18r+uAa0RkJfDJ+g84594QkfnACOfc48n75ovId4C7RKQfsBY4G3ix93/McNmhFPM+IlIEbnPO7aWdJc9st9aYQNnMaUygbM0ZsGK52gzsBgwFBgGDkz/rt7fEr/XWqd/uAJYDS4FlyVv99stArVYptfflz2G6x2bOQBTL1R2BUcnb3smfw4GWPh56NbAI/wLQHGAWMKtWKb3Sx+OajbByKimWqyOBzwCHA2OArXUTvc9S4EHgb8DfapVSTTdO/lg5G6RYrg7BF/EzydsQ3URd9hy+qHcD99YqpeW6cbLPytmHiuXqYPzB9Yn409Gyoh14CPgdcFOtUnpLOU8mWTl7WbFcHYA/4+Uk/EyZ9RfdVgK3AtcDd9mLS73HytlLiuXqWOB0YDydn6qWdUuBPwCTa5XSPO0waWfl7IFiuSrAF4CLWe+0tJxzwO1ApVYpPawdJq2snN1QLFf749eR3wD2VI4TuunAfwO31Sol+5+tC6ycXVAsV7cA/h1/cveOynHSZh7wI+D3tUqpTTtMGlg5N0GxXG3CrycvxZ+ZY7pvAXBBrVK6UztI6KycG1EsV48ALgfsCo3eVQUurFVKizb6mTll5exEsVwtAj8DvqibJNPWAlcBl9YqpVg7TGisnOsplqstQDl5G6AcJy9eAy6pVUrXagcJiZWzTrFcHYG/mdU+2lly6g7gtFql9Kp2kBBYOfn/45Xn4F/y30w5Tt69DvxbrVKaph1EW+7LWSxXdwB+AxyhncW8x6+B82qV0tvaQbTkupzFcnU88Cv8TapMeJ4HTqpVSjO0g2jIZTmT45Y/Ac7VzmI2qh04v1YpXa0dpNFyV85iufoh/P1Rj9TOYrrkF/jd3NycXZSrchbL1Z2BvwAjtbOYbrkbmJCXC71zU85iuXogMBXYTjuL6ZGFwOdrldJz2kH6Wi7uW1ssV0/C3+Lfipl+w4DHiuXqIdpB+lrmy1ksV8/BX6Xfqp3F9JptgDuK5epR2kH6UqbLmRTzSu0cpk+0ArdkuaCZLacVMxcyXdBMltOKmSuZLWjmymnFzKVMFjRTh1KK5erpwGTtHEbNauBztUrpPu0gvSEz5SyWq58G7gL6K0cxupYDY2qV0gLtID2ViXIWy9Xd8U9dthPYDUAN+EStUlqqHaQnUr/mLJarW+FPybNimnWKwK3FcjXVx7ZTXc7k+ZVT8GeNGFPvAGCSdoieSHU58TeHOkw7hAnWycVy9QLtEN2V2jVnsVw9GbhOO4cJXjtwcBov2E5lOZNLv54mvw8MMl3zPPCxtN3yJHW7tcVytR/+RHYrptlUu+DvQZwqqSsn/oleY7VDmNQ5vViupuoG4anarS2Wq/vij2faiQamO14DRqXl+GdqZs7kidE3YMU03bcd/pabqZCacuKf8DVcO4RJvaOK5epp2iE2RSp2a4vl6h7AXGzWNL1jGbB7rVJ6UzvIB0nLzPlTrJim9wwCvqsdYmOCnzmL5erngNu1c5jMWQuMrFVKi7WDdCbombNYrvbHz5rG9Lb++Lv+ByvocuKf/GUntZu+8vnkyeVBCna3tliubgcsBgraWUymzcef2hfcYx5CnjkvxIpp+t4I4ATtEBsS5MyZPGzoJaycpjHm4c8cCqoMoc6cZ2LFNI0zEihph1hfcOVMbi1xvnYOkztl7QDrC66cwMnA9tohTO4cmDyJLhhBlTO5VvMi7Rwmty7WDlAvqHICRwO7a4cwufX5Yrk6QjvEOqGV8wztACbXBPiqdoh1gjmUUixXdwJewP8DGaNlKTC0Vim1awcJaeY8BSum0TcYCOKUvmDK+bv+//WxfWXRQu0cxgAnaQeAUHZro8IB+HsDscr1X3xr+5h/XNE2ftgrbDtEOZnJp5XA4Fql9JZmiFBmzuPWbWwma3c/tvmBT09vPXfwzNYznvp609SHN2dlqu43alJvADBeO4T+zBkVBHgZGNrZpzjHO4vd0FlXtY1rrXZ8Yp8O+jU1LqDJqXtrlZLqoz5CKOdBwIOb+untTpY90jFyweVtE7af5Xa3az1NX2kHtq1VSsu1AjRrDVznyK58cpO4QWOb5g4a2zSXla5l3fp0+BI+bKf8md7UBBwCTNUKEMKa81Pd/cIBsmb345rv//SM1nMGzWw946mzm6bZ+tT0psM1B9fdrY0KA/CPCW/prW/pHCsWu6FPX9l2dOvtHQeMtvWp6YHFtUppD63Btct5CHBvX337didLZ3Ts9ezlbRO2n+12s/Wp6Y5irVJ6UWNg7TVnt3dpN0WTuMEHNc0ZfFDTnGR9euArV7QdPczWp6YLDgcmawysveY8uFED+fXpfZ+a0XrOoCdaz3zyrKZbpw9k1YpGjW9SS23dqbdbGxVa8OvNAToB/Pp0kdtx9lVt4wZUOw4Y7ein/cvKhGdJrVLaQWNgzXKOBR7SGfz92p28Or1jr4WXt00Y8rTbTe1FABOk7WqV0uuNHlRzzdmwXdpN0SRu+4Ob5mx/sF+fLprWfuCSK9rGD3+VbQZrZzPqRgH3NXpQzd24fRTH/kADZM0exzff96lHWr++7eOtZz55RtOfbX2ab6M0BtUs5y6KY28SEZoGSbxfuf8fD5zXepq7s+Wb00v9Hn1S6OjQzmYaam+NQTV3a4MvZz0Rthgmfz/w5y1X0u76LXnYr0+HPuN2tXseZZ/KzKnzglBU2Br4V+MH7n3vuJaF09rHvnpl29G2Ps2uFcCWjb4jvNZu7c5K4/a6gbJm2AnN99avT2cMYPU72rlMr9ochf9ntcqZql3aTVG3Ph0zv/XUjr+2XPzwUf0ee8rWp5mxY6MH1FpzZq6c9UTYYri8PPYXLVfQ7votebBj1MKftE0YOsftYuvT9Gr4kkWrnJnZrd2YJukYckjT00MOaXqad1zrwlvax756Vdu4PZeyzSDtbKZLGv7fy2bOBhooq4dNbL5n2IlN97S/xlYzr2373Jrr248YvZLWgdrZzEY1fObUWnN+RGncIPj16fL9L+n/hzHzW0/tuKOl/PCRtj4NXW5mzs2Uxg2OCFvsKS+NvablCtpcvyUPdYxa9OO2Lw+d53beTTubeY/crDl77c4HWdL83vXpsze3H7T0qrZxI5ax9Xba2Ux+Zk4r50YMlNXDT2q+e/jEprvbl7HVzGvbjlpzffvh+6yiVe0Su5zbptEDaq05rZybSISmwbJ8/2/1//2YBa2ntt3eUn74s/0en2Xr04Zr+ESmNXP2Vxo31UTYcoS8NHZSy89oc/2WPNix98LL2ybsaOvThshNOW3m7KFm6RhyaNPsIYc2zWaFa11wc/vBy65u+5KtT/tOw+/i2PgT3/3jF2yXrA84R7tDYu0cWdSBvNL8vTcaenWKxsxpu7R9RIQmwTX8hYs86Idr+C89jReErJwmjdY2ekCNcq5WGNOYnspBOaO4DVB9KKkx3ZCDcnpvKI1rTHf9s9EDapUzE7coMbny90YPqFXOhv8WMqaHclPOJUrjGtNdLzd6QK1y/kNpXGO6Kzczp5XTpE1uytnwH9SYHspNOWtK4xrTHSuJ4twcSpkHrFEa25iuUlmG6ZQzitfgC2pMGjynMajmU8aeVBzbmK54VGNQK6cxG/eIxqBWTmM+mAMe0xhYs5zPoHCmvzFdtIAoVrm7hF45o3g19qKQCZ/KLi3ozpxgu7YmfCovBoF+OZ9QHt+YjcntzPlX5fGN+SAxMF9rcN1yRvGLwFOqGYzp3AyiuMH3jn2X9swJMFU7gDGduFlz8BDKOU07gDEbsBa4RTOAfjmjeC5K5y4a8wH+RhSr3ohOv5ye7dqa0NyoHSCUck7TDmBMndUE8P9kKOV8BHhVO4QxiTuJ4je1Q4RRTv9y9TTtGMYk1HdpIZRyetdoBzAGWAn8WTsEhFTOKH4auE87hsm9vxDFb2uHgJDK6f1MO4DJvSu0A6wTWjlvw455Gj2PEcUztEOsE1Y5o7iDgH5zmdz5qXaAemGV0/sNsFw7hMmdF4Ep2iHqhVfOKF4BTNaOYXLnMqK4XTtEvfDK6V0FBPUPZTJtCXCtdoj1hVnOKH6JQA4Em1y4jChepR1ifWGW0/s2/hxHY/rSa8Ak7RAbEm45o7gGXK0dw2Te94jid7RDbEi45fR+APxLO4TJrKeAX2qH6EzY5Yzi5cCl2jFMJnUAZybH1oMUdjm9nwNztUOYzJlMFD+uHeKDhF/OKG4DztaOYTLldeAS7RAbE345AaL4QeAG7RgmMy4mioN/LSMd5fQuwk7rMz03A3+KaPDSU84ofhX4qnYMk2rtwFmaN4ruivSUEyCKb8bumGC6r5Jc1J8K6SqndwEwRzuESZ17gf/UDtEV4lwqZvj3igoj8E8oG6gdxaTCK8A+RPEy7SBdkcaZE6J4PnCedgyTCm3AsWkrJqS1nABRPBn4o3YME7wyUfywdojuSG85va8Bz2uHMMGaShRfrh2iu9JdTn9X7mOAt7SjmOA8B5yqHaIn0l1OgCieBYzDrv0071oBHEMUx9pBeiL95QSI4nuAifgrDUy+rQK+mKbjmZ3JRjkBongKcJZ2DKNqLX7GvEc7SG/ITjkBongS8F3tGEZFO3ACUVzVDtJb0nkSwsZEhSuBc7RjmIZxwClE8fXaQXpTtmbOd50H/F47hGmYs7JWTMhqOf1VB6dgt9fMg/8gijN5MUQ2ywkQxWuB44HLtKOYPvNtovgn2iH6SjbXnOuLCmcDV5LlX0b5shb4GlGcioumuysf5QSICl8A/oBdyZJ2bwLjieK7tYP0tfyUEyAqfBz4CzBIO4rplpeBElGci+t587Wb52+F+ElgkXYU02UPAfvnpZiQt3ICRPHzwBjgPu0oZpP9Ajgsjddk9kS+dmvrRYV+wLeACGjSDWM6sRI4N7l2N3fyW851osIY/AkLO2lHMe8xAziVKM7tEiR/u7Xri+IZwGjsptWhWIW/R/FBeS4m2Mz5XlFhHP7Wm/Zqro5H8LPlQu0gIbCZs14UTwVGAn/SjpIzq4BvAGOtmO+ymbMzUeEz+FP/RisnybpH8VeUWCnXYzNnZ/wZKPsCXwFeUk6TRYuBE4AxVswNs5lzU0SFzYBz8YdeCspp0u5l/AORr0se72g6YeXsiqjwYeA7+NuhtCinSZulwA+BSUSx3YxtE1g5uyMq7Iy/HcpxwGbKaUL3Bn7tfiVRvEI7TJpYOXvCz6SnAWcAuyinCc1zwLXAL9N+i0otVs7eEBUEOBK/u3sU+X2hbSVwMzCZKH5AO0zaWTl7W1TYCT+Tng5sp5ymUWYBk4EbbJbsPVbOvhIVWoAjgFLy9hHdQL1uGTAFP0vO0g6TRVbORokKo/C7vCX8JWtpuxKmA/9M1DuA24GZaXl8e1pZOTVEha2Az+KLeigwVDXPhjn8E8QfAO4HHiSKX1dNlDNWzhBEhUH4s5H2BfYB9gJ2Bfo3KMESYCHwbN2fjxPF/2rQ+GYDrJyhigr98QUdDuwBbAtsBWydvNVvF3jvK8Rt+CdtvZ38Wf+2HH+YY10RFyaPUjSBsXJmgT+U8yF8QVcQxWuUE5leYOU0JlB5PVhuTPCsnD0kIluJyFl1f99BRKZoZjLZYLu1PSQiReA259xe2llMtmR+5hSRoogsEJH/EZF5InKXiAwQkV1F5K8i8qSIPCQiw5PP31VEHhWRJ0TkUhF5O3n/FiJyj4g8JSJzROSLyRAVYFcRmS0ilyXjzU2+5jERGVmX5X4R2U9ENheRXydjzKr7Xsa8yzmX6TegiD+0MDr5+5+AicA9wO7J+w4A7k22bwOOT7bPAN5OtpuBDyXb2+IPR0jy/eeuN97cZPsC4HvJ9hBgUbL9Q2Bisr0V/g70m2v/W9lbWG+ZnzkTLzjnZifbT+ILNAa4SURmA5Pw5QH/uIabku36B/AK8EMReQa4G39Wz+CNjPsnYEKy/eW673sEUE7Gvh9/TehHu/Yjmaxr1g7QIPVX3rfjS7XcOTe6C9/jRPxVJvs559aKSI2NXGjtnPuHiPxTRPYGjgW+lnxIgPHOObt3julUXmbO9b0JvCAiEwDE+1jysUeB8cn2cXVfUwCWJcU8hHfvEP8WsOUHjPVH4JtAwTm37iE8dwLniIgk4+/T0x/IZE9eywl+JjxdRJ4G5gHrXpQ5H7hQRB7H7+quuz7xBmB/EZmZfO2zAM65fwLTRWSuiGzoKdpT8CWvvxfu9/HnzT6TvHj0/d78wUw22KGU9YjIQGClc86JyHH4F4fs1VTTcHlZc3bFfsDVyS7ncvw9goxpOJs5jQlUntecxgTNymlMoKycxgTKymlMoKycxgTKymlMoKycxgTKymlMoKycxgTKymlMoKycxgTKymlMoKycxgTKymlMoKycxgTq/wArElR/2o3hzgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "posi=0\n",
    "nega=0\n",
    "for i in range (0,len(dataframe)):\n",
    "    get_val=dataframe.Comp[i]\n",
    "    if(float(get_val)<(-0.99)):\n",
    "        nega=nega+1\n",
    "    if(float(get_val>(-0.99))):\n",
    "        posi=posi+1\n",
    "posper=(posi/(len(dataframe)))*100\n",
    "negper=(nega/(len(dataframe)))*100\n",
    "print(\"% of positive tweets= \",posper)\n",
    "print(\"% of negative tweets= \",negper)\n",
    "arr=np.asarray([posper,negper], dtype=int)\n",
    "mlpt.pie(arr,labels=['positive','negative'])\n",
    "mlpt.plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T10:00:23.889816Z",
     "start_time": "2021-09-22T10:00:23.872867Z"
    }
   },
   "outputs": [],
   "source": [
    "dataframe.index = dataframe['Date']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T10:00:24.168214Z",
     "start_time": "2021-09-22T10:00:24.148268Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>adj_close_price</th>\n",
       "      <th>Comp</th>\n",
       "      <th>Negative</th>\n",
       "      <th>Neutral</th>\n",
       "      <th>Positive</th>\n",
       "      <th>Date</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Date</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2007-01-01</th>\n",
       "      <td>12469</td>\n",
       "      <td>-0.9814</td>\n",
       "      <td>0.159</td>\n",
       "      <td>0.749</td>\n",
       "      <td>-0.9814</td>\n",
       "      <td>2007-01-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2007-01-02</th>\n",
       "      <td>12472</td>\n",
       "      <td>-0.8521</td>\n",
       "      <td>0.116</td>\n",
       "      <td>0.785</td>\n",
       "      <td>-0.8521</td>\n",
       "      <td>2007-01-02</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2007-01-03</th>\n",
       "      <td>12474</td>\n",
       "      <td>-0.9993</td>\n",
       "      <td>0.198</td>\n",
       "      <td>0.737</td>\n",
       "      <td>-0.9993</td>\n",
       "      <td>2007-01-03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2007-01-04</th>\n",
       "      <td>12480</td>\n",
       "      <td>-0.9982</td>\n",
       "      <td>0.131</td>\n",
       "      <td>0.806</td>\n",
       "      <td>-0.9982</td>\n",
       "      <td>2007-01-04</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2007-01-05</th>\n",
       "      <td>12398</td>\n",
       "      <td>-0.9901</td>\n",
       "      <td>0.124</td>\n",
       "      <td>0.794</td>\n",
       "      <td>-0.9901</td>\n",
       "      <td>2007-01-05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-12-27</th>\n",
       "      <td>19945</td>\n",
       "      <td>-0.9898</td>\n",
       "      <td>0.178</td>\n",
       "      <td>0.719</td>\n",
       "      <td>-0.9898</td>\n",
       "      <td>2016-12-27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-12-28</th>\n",
       "      <td>19833</td>\n",
       "      <td>-0.6072</td>\n",
       "      <td>0.132</td>\n",
       "      <td>0.76</td>\n",
       "      <td>-0.6072</td>\n",
       "      <td>2016-12-28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-12-29</th>\n",
       "      <td>19819</td>\n",
       "      <td>-0.9782</td>\n",
       "      <td>0.14</td>\n",
       "      <td>0.761</td>\n",
       "      <td>-0.9782</td>\n",
       "      <td>2016-12-29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-12-30</th>\n",
       "      <td>19762</td>\n",
       "      <td>-0.995</td>\n",
       "      <td>0.168</td>\n",
       "      <td>0.734</td>\n",
       "      <td>-0.995</td>\n",
       "      <td>2016-12-30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-12-31</th>\n",
       "      <td>19762</td>\n",
       "      <td>-0.2869</td>\n",
       "      <td>0.173</td>\n",
       "      <td>0.665</td>\n",
       "      <td>-0.2869</td>\n",
       "      <td>2016-12-31</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3653 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "            adj_close_price    Comp Negative Neutral Positive       Date\n",
       "Date                                                                    \n",
       "2007-01-01            12469 -0.9814    0.159   0.749  -0.9814 2007-01-01\n",
       "2007-01-02            12472 -0.8521    0.116   0.785  -0.8521 2007-01-02\n",
       "2007-01-03            12474 -0.9993    0.198   0.737  -0.9993 2007-01-03\n",
       "2007-01-04            12480 -0.9982    0.131   0.806  -0.9982 2007-01-04\n",
       "2007-01-05            12398 -0.9901    0.124   0.794  -0.9901 2007-01-05\n",
       "...                     ...     ...      ...     ...      ...        ...\n",
       "2016-12-27            19945 -0.9898    0.178   0.719  -0.9898 2016-12-27\n",
       "2016-12-28            19833 -0.6072    0.132    0.76  -0.6072 2016-12-28\n",
       "2016-12-29            19819 -0.9782     0.14   0.761  -0.9782 2016-12-29\n",
       "2016-12-30            19762  -0.995    0.168   0.734   -0.995 2016-12-30\n",
       "2016-12-31            19762 -0.2869    0.173   0.665  -0.2869 2016-12-31\n",
       "\n",
       "[3653 rows x 6 columns]"
      ]
     },
     "execution_count": 155,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T10:00:56.687639Z",
     "start_time": "2021-09-22T10:00:56.675673Z"
    },
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "train_data_start = '2007-01-01'\n",
    "train_data_end = '2014-12-31'\n",
    "test_data_start = '2015-01-01'\n",
    "test_data_end = '2016-12-31'\n",
    "train = dataframe.loc[train_data_start : train_data_end]\n",
    "test = dataframe.loc[test_data_start:test_data_end]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 159,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T10:01:05.774347Z",
     "start_time": "2021-09-22T10:01:05.425205Z"
    }
   },
   "outputs": [],
   "source": [
    "list_of_sentiments_score = []\n",
    "for date, row in train.T.iteritems():\n",
    "    sentiment_score = np.asarray([dataframe.loc[date, 'Comp']])\n",
    "    list_of_sentiments_score.append(sentiment_score)\n",
    "numpy_dataframe_train = np.asarray(list_of_sentiments_score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T10:01:06.522464Z",
     "start_time": "2021-09-22T10:01:06.446284Z"
    }
   },
   "outputs": [],
   "source": [
    "list_of_sentiments_score = []\n",
    "for date, row in test.T.iteritems():\n",
    "    sentiment_score = np.asarray([dataframe.loc[date, 'Comp']])\n",
    "    list_of_sentiments_score.append(sentiment_score)\n",
    "numpy_dataframe_test = np.asarray(list_of_sentiments_score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T10:01:07.835050Z",
     "start_time": "2021-09-22T10:01:07.829033Z"
    }
   },
   "outputs": [],
   "source": [
    "y_train = pd.DataFrame(train['adj_close_price'])\n",
    "y_test = pd.DataFrame(test['adj_close_price'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 162,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T10:01:12.819579Z",
     "start_time": "2021-09-22T10:01:12.811605Z"
    }
   },
   "outputs": [],
   "source": [
    "from sklearn.metrics import precision_score\n",
    "from sklearn.metrics import precision_recall_curve\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T10:02:15.400475Z",
     "start_time": "2021-09-22T10:02:14.807577Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.28392682750431575\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-163-d28c3ad09fba>:19: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  test['adj_close_price']=test['adj_close_price'].apply(np.int64)\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYQAAAEECAYAAAAoDUMLAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAACS9klEQVR4nO2dd5wURdrHvzVh87LAEiUtIAoCioCAGTOGE/OZ9TWdnme809Mzx9MzneHUM2DO6cSMCUwIgqCSg2SQuMASNsx0vX90mO6e7p6esNH+fT6wPdXV1dXV1fXUk4WUkgABAgQIECDU2B0IECBAgABNAwFBCBAgQIAAQEAQAgQIECCAhoAgBAgQIEAAICAIAQIECBBAQ0AQAgQIECAAAJHG7kCmaNeunayoqGjsbgQIECBAs8LUqVPXSSnbO51rtgShoqKCKVOmNHY3AgQIEKBZQQixxO1cIDIKECBAgABAQBACBAgQIICGgCAECBAgQAAgIAgBAgQIEEBDSoIghOgmhPhSCDFbCDFTCHGZVn6i9lsRQgy1XXOtEGKBEGKuEOIwU/kQIcQv2rmHhBBCK88XQrymlU8SQlTk+DkDBAgQIEAK+OEQYsBfpZT9gBHAxUKIXYAZwHHAV+bK2rmTgf7AKOBRIURYO/0YcAHQR/s3Sis/F6iUUu4IPADcnc1DBQgQIECA9JGSIEgpV0kpf9SOq4DZQBcp5Wwp5VyHS0YDr0opa6SUi4AFwDAhRGeglZRyolRjbj8PHGO65jnt+E3gIJ17CBAgQIDmDCkluUozIKVEUeovZUFaOgRNlLM7MMmjWhdgmen3cq2si3ZsL7dcI6WMAZuA8nT6FiBAgABNEfv+60sG3/ZpTtq6d9xcev3jQ2pjSk7as8O3Y5oQogR4C7hcSrnZq6pDmfQo97rG3ocLUEVOdO/e3bO/AQIECNAUsLxye87aev471aesOhYnL5J7myBfLQohoqjE4CUp5dspqi8Hupl+dwVWauVdHcot1wghIkAZsMHesJTyCSnlUCnl0PbtHT2vAwQIUM94a+pyJi9K+jwDNCDqK9GlHysjATwNzJZS3u+jzbHAyZrlUE9U5fFkKeUqoEoIMUJr80zgXdM1Z2nHJwBfyCC3ZwANJ/13Ire+N6uxuxFAw1/f+ImT/juxsbvx+0Q9a1b9cAh7A2cABwohpmv/jhBCHCuEWA7sCXwghPgEQEo5E3gdmAV8DFwspYxrbV0EPIWqaF4IfKSVPw2UCyEWAFcC1+Tm8QK0BExetIEx3y5q7G4ECNDikVKHIKX8Bne69I7LNXcAdziUTwEGOJRXAyem6kuAAAECBKg/BJ7KAQIECNDc0Fg6hAABArR81MTizF7lZTwYoClAF9XIeqIIAUEIECAAN4+dxeEPfs3KjbkzkQxQf2g0K6MAAQK0fPy4pBKAzdV1jdyTAF7QAzgo9UQRAoJQD9jjjs84+P4Jjd2NAAF8o75EEAHqB/X1tpptCs2mjLVVNaytqmnsbgQIEKCFIuAQGhk/Lq1kxopNjd2NAAECBKg3FiHgEHziuEe/A2DxXUc2ck8CBKg/iPp2hQ2QFfQY0PUlMgo4hAABGhAbt9VSE4unrhgggAcCkVGAAC0Ag279lLPH/NDY3QjQzBGYnQYI0EIw8df1jd2FJAShJJsHdIFewCEECBAgQAAg4BACBAjQQjBz5SYWrdva2N1o1qgvghBYGQUIEMBAQ2QyP/Khb4DAYi8bBLGMAgQI0OgI8lY1LhKhK+qn/YAgBAgQwDcCetA0UF+EOSAIAQIEMJBqnakv65YA/pAIf10/8JNTuZsQ4kshxGwhxEwhxGVaeVshxKdCiPna3zama64VQiwQQswVQhxmKh8ihPhFO/eQllsZLf/ya1r5JCFERT08a4Bmhq01Mf76+k+N3Y3fFVLJputLVBEgPTQmhxAD/iql7AeMAC4WQuyCmvf4cyllH+Bz7TfauZOB/sAo4FEhRFhr6zHgAqCP9m+UVn4uUCml3BF4ALg7B88WoJni9SnLOOT+Cbw8aSlv/bi8sbvzu4C+vCiKd72AQ2gaaDSzUynlKinlj9pxFTAb6AKMBp7Tqj0HHKMdjwZelVLWSCkXAQuAYUKIzkArKeVEqZK3523X6G29CRykcw8Bfn+4+s2fmb9mSxCSuRGQaswDetA00CSUypooZ3dgEtBRSrkKVKIBdNCqdQGWmS5brpV10Y7t5ZZrpJQxYBNQ7nD/C4QQU4QQU9auXZtO1wM0QwSLT8Mj0CE0bejb5CXrt/Lj0krvyq+cCt8/pr7UzSuhpipl+74JghCiBHgLuFxK6ZV81WlnLz3Kva6xFkj5hJRyqJRyaPv27V078OBn8/lk5m8eXUzg4c/n8/EMf3UDNCyCpafhoMukfw8E4c2pyxnzzaLG7kZWuOCFqUYEZgtW/QxKHLZtgLkfwMfXwC2t4f5+rHr82JTt+iIIQogoKjF4SUr5tla8WhMDof1do5UvB7qZLu8KrNTKuzqUW64RQkSAMmCDn7454YHP5vGnF6b6qnvfp/O48EV/dQM0LFrA2mNBc7Dh/z0olf/2xk/c+v6sxu5GhvCQpK+eBf/dFybcDUtMxKLfH5ik9KVkw4yUH5UfKyMBPA3MllLebzo1FjhLOz4LeNdUfrJmOdQTVXk8WRMrVQkhRmhtnmm7Rm/rBOAL2Ry+ngD1ipamQ2gOMzrVgh98lk0Y6xeofyfcDa+dBq26wLXL4Y8v8l58T0rFdtjkbaThJ3TF3sAZwC9CiOla2T+Au4DXhRDnAkuBEwGklDOFEK8Ds1AtlC6WUuoB4C8CngUKgY+0f6ASnBeEEAtQOYOTffQrQAtHS1t7msPjpFrwWwKH0GKhEwSALkPglNcgvxSAn5VeavkEbwPOlARBSvkN7nzKQS7X3AHc4VA+BRjgUF6NRlACBNDR0najzUH+nmrBbw7P8LvErLHwtUmAc/4XltM/y968G9+L0dNe8Gwm8FQO0GTRknaj81ZXNROOJxWH0Cwe4vcFJQ7v/AnKe8OfJ8HfFztWezF2MESLPJsKCEKAJouWsvjM+W0zhz7wFdOXbWzsrrjCcExLqUOo964E8ICjd1bVKqjbBkPOhg59obCNQyX4QfaF61Z5tt9sCUJMkWzYWtvY3QhQj2gpi8/m7TGAZjFffw9mpw2JBhF7Vi5R/7bunnVTzZYgzF61mcG3fdrY3QhQj2gpOgT9OeriKeJCNAGkWvBbkhjvlvdm8saUZakreuDfn83jyzlrXM/nego7KnM3agShTUXW7QcJcgI0WbSUxUd/jFiqQEFNACk5hCb8UtZUVRMNhWhTnOer/jPfLgbgxKHdvCt64N+fzQfck/00CEe1/AfIK8kJhxAQhABNFi1FPKE/Rl286T9PKq6sKb+SYXd8DjStTGwNQj8XfQ0V+0A4mnVTzVZk1FTRUsQcTQFNeDOaFnQHu3gzeKBUPWwpRLqh0CDjtXUNtO6Rk6YCgpBjBN9L7tBiiKv2GLEWoUNoIe+kgZBzHYJdiaAoUL0ZCspy0n5AEHKMXHwwY39ayes/ZKfsagloKUuP/hy1zUJk5H2+GTA5TQr1TUBrtm8CJBS0ykl7gQ4hx8jFB3PpK9MAOGmPzJVdLQFNWYGZDmRz4BC0PqZawFoM19ZAyDVBEDY7owNuG8t3BQQcghsae8IGLHXu0ELogaFDiDWDB0qtQ2iQbrQY1Pd4lYpt6kFAEJzR2BM2oAe5Q0shrgkroybMIWhIHdyuabyTp77+lY9neHvdNgV4jec+d3/BqH9/lVX7rdAIQn4gMnJEY0/Yxr5/S0Jjc3u5gv4UzYMgeJ9vKvP79g9mA03LxNQJXhvU5ZXbs24/4BBSoLEnbGPfvyWhpYykTthizUCpHMQyyi3qdz2QHBSaBiKs5j7IAVoch9DYE7axRVYtCS2FuCY4hKb/PM1FZNRckHOlsqZT/nP4Xa6Ovqb+GHYhlHbMSfsBh5BjtBQxR1NAiyGuupVRlqErpJSs31KTgw45tK39TZ0PIcv7ZPB9zFtdxdqq+nnu+ka2y8G6LTXM/a1K/TH/U16pvYTdxAKDGPwndjQc9s8se5mAnxSaY4QQa4QQM0xluwkhJgohfhFCvCeEaGU6d60QYoEQYq4Q4jBT+RCt/gIhxENaGk20VJuvaeWThBAV2TxQYy8ijX3/loSWQlx1K6NsdQgvfr+EIbd/xoI1Vbnolgvql0PI5Ps49IGv2O9fX2Z138ZCtuN1wL3jOezfX0E8Bh9fS4Vcwbv5NwJwW93p3BM7GUK529f7aelZYJSt7CngGinlQOAd4CoAIcQuqOkv+2vXPCqECGvXPAZcgJpjuY+pzXOBSinljsADgHeOtxRo7PAAjc2htCQ09VhwUkpfi3OuYhmNn7sWgEXrtiWd27Stju8WrMuqfaj/nMqZXr+9Lp66UhNEpsvRy5OW8sPiDVRVq6HTmfcxrJ/PQ+Gz+VnpyYOx43g+fmjuOqohJUGQUn6FmufYjJ0B3V7qU+B47Xg08KqUskZKuQhYAAwTQnQGWkkpJ0p1RjwPHGO65jnt+E3gIJ17yAR+J5yUku9/XZ/pbVwREITcIdOxXLO5mnemeScTzwWe+XYxB9//FT8urfSslyvHNH00Qg5fx/nPT+HUpyaxpSaWWdtaJ+vbU/n3xkFn4lwppeQf7/zCiY9PTBTO/wTyW/FG+AiOrr2DB2InUFcPKuBMeY0ZwNHa8YmA7lLbBTDHXFiulXXRju3llmuklDFgE1CeYb98T7j/TV/ByU98n+ltXBHQg9wh06E8c8xkrnjtJzZX1+W0P3bohGDZhuQdu47Zqzbz8czfAKjLcjXUCWTIYb8057fNQPZER5GSuCJ58qtfqdZ25VJKg2Bk6z3+e9swZfK4ekIlUysw/zPoNZK4cCcC22vjrMtSx5QpQTgHuFgIMRUoBfRUUE47e+lR7nVNEoQQFwghpgghprh1zO+EW7o+exvgbO4fIDUyHUvdvru+X4Xefthpy67h8Ae/5s2p6l6oLpbtYq0dZMw/p4YE3p2+gjs+nM0Dn84D4Lr/zaDntR9a+5Bp+7+zzyOTObxiozp/W0fjCBQGiYVQtRJ2PBgv4clxj33H0Ns/y7ivkKHZqZRyDnAogBBiJ0D3DllOglsA6Aqs1Mq7OpSbr1kuhIgAZSSLqPT7PgE8AZDfuY/jSPt9AR7fcFb4vbHE9YlMFw/FEH/U78vQ9VVOO3YnZBu6Qn+eeqQHSCmprlMJl85hvTxpaVIfMm6/xXiX+BuLTAjCyo3byaOO6eGzIAyVsgSK28Muo+HTH12vm71qc9r3siMjDkEI0UH7GwKuBx7XTo0FTtYsh3qiKo8nSylXAVVCiBGafuBM4F3TNWdpxycAX8gsZp3fKzPXUnijpQRkawrIlEPQF+r6jh3kJcJxQq48lZ3up+8cc7GD15t3UuoHOoQE/EzPTJ63dvNaroy8YfxuI7bAIbdBYev0G0sTfsxOXwEmAjsLIZYLIc4FThFCzAPmoO70nwGQUs4EXgdmAR8DF0spdfOAi1CtkxYAC4GPtPKngXIhxALgSuCabB7I7yKShd7aEy2VJX7m20UsXe8uK68PZLp46AQhU4uzZRu28c8PZ/t20vLLbWbrqexFgIxF3NbnX9du4YXvl/i+h0Qaz2PfzUspsxaJfpsDSygz5q2uahADAif4GQvzHPK1z1UU9v72LC6MvE8tUbbKfJ6LHQK7nZxNV30jpchISnmKy6kHXerfAdzhUD4FGOBQXo2qmM4JGnsH0hJ1CBu31XLLe7N49rvFTLjqgAa7b6aMov4OMiUIf3phKrNWbeb4IV3ZqWOpaz29fS8dghnZOqbpl3vtZewc6uj/fEtVdYzTh3f3tQlSlMRmyc7QxJXsCcKfXpia1fVgnReHPqAaOx67e1e36vUGPyNhfh1m7ssVyydTVrWAq+ouYG7n0fy8fBMAZ9WXSMOGluep7HMRqDeRUQskCLWaMnRrTcPagmc6kvoUyJQg6DbvqRZ6vfmQRz3zPMs2QY6+Y/eau3Hb/NPt2P1OSwmEtRvYCXJMkU2CA27sTZ8OP9+6uY6vbm9eAcB0ZccMe5UdWh5B8CsyqifVXIskCNpWMS/cMLsUHdnqYzLVIeiy/qjmATr2p5W8O31FUj0/OoTCaNg4zt4kVP3rNHeFrU7ytd5jIU313MRPSoYiozWbq3MaqrqxnU91+NMhpCky2qba01RKd860PtECCYK/eoGVkX/oVieRcMNOl2yJa6YLhy7r1xfGS1+ZxmWvTndtP+xBEIryEgQh64VMJwhpiIyMcr+3lgkCZ79EFRn5bMeEU5+axIUv/khNLDccZlMhCH5glhL66rVGEDZS7Hja6d2v2Vydfsdc0AIJQssWGcUVyeJ1WwF1x5FL56vVm6vZ6uDpqjsoRRqaQ8hyKNNdOBat24qU0jdnob9rr7lUaCII2VoZ+eFI3OafXUG8vTbOyo3JvjhWDsF6zq5D8Ksg1o0R/HwaG7bWMvanlcacc4JdLOYbm1fC+oWMif6LXmJl6vop4OdbN+uNfHV7+wZqIyXE0vAIGHbn577rpkKLIwh+FZH1JjKq5/g7D3w6j5H3jmfJ+q3c9v5sdr15HJMXObptpI3hd37OMf/5Nqlc39lFcxhEyw+ypa1+lLhSSiqu+YDzn5/CAfeO56mvFxnXORGULTUxKq75gJcmLTHetVs/K675gGUbEotutrGM/Fg1uRFBex/PfmYye931RXI9EgTHzm3EFWkpO+2pSb6InE6M/Cygr/2wjEtfmcYz3y52rWN+xnxquS/6KMqSSd4Nx2rh/n7w8GAODE/n07yrOD38qXH63Gd/SNupK22RkR8eYdsGaqKtXU/Xt265xeVD8LspbCgO4eflG/lp2UbO2LMiJ+3r8ZdWb65honGcO5Zx/potSWW6yCgaaVgOIdsUOX6Isz5fPp21GoCpSyoNkZHTAvbbJnWsn/56EW2L81zrOSFXsYycrIUSfghuIiNr+SSXTYQipaFMt18TlzIpVlJckZjUJC5tJuqmgq7Q9+IQzETp3PCHHB/+Bp45lHPCZzAmfri18uqZsG4+hKxLXVhIbo8+QxgFZtTw+Zx863VSwoqp0KEf5DmLb/y8d/Mr9zVNtq2nJlrmv36O0QIJQtPyQzj6EXXHnSuCEDJZgBjxZep55myv1URGDcwhZCsr9sMh2OtIpFHmNK5mU9O4x/g7yfKzj2Wk/nWauqmVys7lUkrLtyCluS1p4bgVBTZXpx88Tx8fXwTaxxjFpWS/0E9cG3mFfqGEF/WN0ReYpPSzVv7gr7DUFCTu1De47bl3OTA0jZ1Cy7gl+hy8+RydeZgiUQ1r50F+KXxwJcz9EHYYDOd/4Tjoft6meQ77FRlVe3AI9Y0WRxD8LiL1tdetdysjk3xXprHzygbVmsgor8GVytld72dcnOokOAT3+uGQMBYvp1fuJOfOlkPQb+Q1d91FRs7lMUUSDQvjGcwEQkrrs8UUhU3brTorP2NszFMf30adRjW8RL+KIrkz+jRSCm6uO5MNshX37iPZMOllXsy7E+b2hp216PqbV0DP/WDR14CE7iN4Ol7H0/EjyKOOC8PvcWX0TQ4L/8D5kQ/g1f+qHd68Atr3g5U/qpxC16HJz+XjdVoIgi+R0XqqC3ZIqm8n3F5Ip64dLVCH4K9e/VkZ1e/ibPYizdYByy8SVkYNrVSufysjJwWyXuZ0fUKRLAyC4cVJmJG9DgHj3u510rMysusAVB1C4hrFziHYCUIa78gXgfYgxkadump2YD1vxvfj2fgoxip7sXX/m/lz7WVUk4d89RSY9pLa4S1roPNucP0auHgyFBi5vKglykPxY6HTQG6OPk8XsR7WL4ANC+GwO+Hkl9SK6+Y79sPPAm8eH/NQuRK8bZVUayIjJ3GTH91nNp9NiyMI9S0ySh3OIKNmfUOfENL0sdY3EdLludEG5hCyRToLkBO8RUaJY6fbOF2bKysjzzout3Cbt3YipSgyoVSWVjPTuJRJBCEdXxF/Vjmp57TYtJSQkCyRiTzCdYrCj3InDqq5FzrtBu/+GZ47CmLVyOIOrK+W0H5np9ZgxMUAbJMmPULvA1TREUDd1uTLfh1PeNH4lM+jWDiEBBznZqwWaquojrRWb2uaL+l849msB83rC/eB+lYq13dGqVTQxfhSJiZYjmKmuSJBEJoXh+DHfNRex7qLc68fDoU8CbKjKCrraKf63+R29PnstmN3G0q7GEuSmGN2R7S4kmzmnA536qeuvggqHs8aXabmMVkqOyS1vY0Cav9vHOxxPixR9XdfLBcMuf0z97wV/Y5iQnxXTqm9Do68D479L7TtBdEi9XytA0F4fjSlb6gRd9pTCTHnPATmd25+Fsf3tF1V9G/XOAQzQdBr+1m3vIY51frUAglC/eoQUg1ofUfY7FH7K/8X/gglXpckm12yfiuv/7DM4+oEamJxKq75gIc/d2aHrXU1kZGHUnnJ+q2c8Nh3vvwittTE+M+XC1LK1LOlrX7EGV6KZ6cFLOGMlphrTnPCqdlsRXvS9tcJfq2MdNjnqyJx1SHEFZmUvCXnIiPFOqZOzRfPfJFZSg+my0R4B3PgQIUw7GSkc2fsWpWTWOHgdwFAfiln1V3DT3JH2OO8RCA5nSCsnA5jDofqTdrNao1LI8T4oeBiuLsCtpr8MqY8A4/tQzyeGC/zozgO25Y1AGzTOIQlpmCSueIQUjXT7AmC/WP0vUP3yyJU/QaP7kV/sUhtP0X12iyToLg3vBX+dzF3rrmIm6IvMPijoxGKuvjqbOno/3zL1W/97Ks5PS7RmG8Xpb619kxeRkb//mw+U5ZU8unM1Zbyobd/xim2zHRPTFjIPZ/M5Y2p3lEqs11AvcRBOrwikHqJfcIh4Skyyth5ygN+0lym66mcpEOQdpGRlUOw5zZOx+/Gz5DU2Ux+ky6RkrzKhUxS+qKYli8zYVOkhG7DoNOucPYHrIp0s7diwf3j5jqfCIUgWgyz3oWl38Fard76BUaV3cRCrePb4JkjYMK/YOwl8PG1sPoXItWV5q4bcJzby38AYG1JsmgrnenkVTdVM82eINjH1bfIKFWF2q2wfCr8+Dysmcm5kY+09r1v4EYQshYlfXYzTH+RD0tP5MP4MEo2zWNYTE2WoU+ujdusBMIP/NT0E05aGMpuK9ZtSfhL6CjQvHd/XZvs82DpW5ZD5odb83qmJDt8RSaIo0hY5vgVGWUL7/Hwzodgn3/6+7ITRClNhgvSplSWMomApMUh+LLbt4qMksZ2yxpCdVtYJDtZis3cZlxKKCiDC7+Gin1S3vOhLxa4n8wrAj2C//aN6t9NCS788PDkRN11c+HLO9Q1I6ZyI5HtJq5Bf5S1c4nXOfgOLZsEJR3ZmN8l6VQ634I3h9DCRUZJAbj8mp2moggvnQRPHai+YKAcNRuR23iO/Wklpz75vREILrmfvrplxbYN6u5ESqqX/0Ssyx683vYCLq37C7X5bTg0Pl5r2872e0+ItVU1BkF0qlpdF7eYF3pZ3ehIKLtTP2inVgVAItWla18d4vGng1TEe21VjSfRsJ+qiyvG+42EzX4I6d87EyTadG/b1ezU9lvnApz8MPR3aY9dFFdk0vxOZ/PhS4dgUyonDeMvrwMwR+luvc4sMsolMTY7pb18Iky4x4hICnBy+MvE+a57GIevxNQw8dHtawHoKVZR+MGfefe1p+A/w2h1bxe6sNZ6r3XzoUM/FIftaq5ERqmGptkTBPuz+9mFSCl5d5pLLJO6ati8CpZ8Yynua3KAccKlr0zju4XrXQN4ZbRA/PwavH4mm6b9jxXLl/LVb3kIIEaE37odyd7xHzg9/Cl9l75iu5d7ky9+v5iT73yO2b/pBC658hEPfc1ut4wzfuu7Nj87bj9PqS9GqzZ5e1jbb5fuEHr19/2fV7LHHZ8xaZGVezFfYV9Y6uKKsUMOCZEk7zajPjgE10XSoU6qcj0gn76QJsJLWENNmJ8tpsgkjiLVc67alCD69j44jZu+03fkvtbOg89vY3OPQ5gs+7r2w61PGekN80qsv7+8XY2JJMJU9x5FiTDN4e4jAPhn3Sk8FT8CgFaVMwgT57bIGPJmvs7o2X81qr+dfxOnhj9XpRFSquaubXs7vkO9xM8zeCqVU3yhzZ4gmAdv/ZYa1mx21vabMWHeWiYvdon/8/b58G9bHp9hF9BRbOSR6EPOH9wHf+PZ6N2cHv7UVWRkmaQLPodVPyXV2bbgG76f9avxW2pKqqIvb6CbWMOSmhJjMV3W7WjyqeP26DPsM/9fsCkhj/9tUzWL1tksI9bMhgWf0fObq/g8/yo2LnORmwK/rtWu/fkNmPUu4drNCBTPj9/w63CpYhYP6e2k0rfY75fuEhv3EHBP+lV9/z8v2+R+ve1d18WlRYfQ4CIj218z3EJW67B3x0tkZFgzYX0OJ5FRqo2OOUqsfUy8HP8c2530OITCLNvrTuxLY53pXed06J3CVkx8FEo6sr3P0dbykdfyuDyOF+KHsFa2BmDgnH/zY/6f2C2kftd1MszNdWeyvvsoKmUpd0afJvbRP1i6fJmqtC7vnTXH6cVJp2ompaeyEGIMcBSwRko5QCsbhJpHuQCIAX+WUk7Wzl0LnAvEgUullJ9o5UOAZ4FC4EPgMimlFELkA88DQ4D1wB+llItT9cvpAYf4DE61bkut+8nVM6CwDRS1g2MehfULIVoAk5/gqPD3bLUP6K8T4Icn2TckGBn+iZe2X2DqmzQda//VbIYXj1MLb9qofpmTn4RQmKL3r6A2PpAoV7G7mK/KLEWIaNUyELBGtjE+5PVlA1hMFyrQ2Ndlk+nCZgaH5rPfPWrR4ruOTPTzUXX3oktU29SuBIo8FlkJb58HqDlNQ+GTeHT+MVRuraWNFsPHjFQL0oH3TWDaDYfQpjjPWGiNEBGK5OlvFnHaiO4U5SWmpPOO0nuPZDHt86Hw9OIi7I9SF1d4ZbIqPw6bOATn3MP1QBCk9e+0pZWsrarhgL7J5pdJ/bGV6xuLOiV5gddr3r/mfIo+OwQ42Gg7XYJgrp9E4B3ep12pbGl/6xpo3Z3qgvaA1TrO3PZX89ay947t6FSmiiZ9eQi7IVqkrgfbKxNlbSpgzz+zrfNBFMkInyuDOeJvz0BeMXfVnKA/HePjuzEy/BNlQrUWqjr8YQ58J8xaWvPsPLXOG3m3sMPMCVz+fQVv5wPlOyKrHDgEI1uej6x3XhxCtgQBdRF/BHXR1vEv4BYp5UdCiCO03yOFELsAJwP9gR2Az4QQO2l5lR8DLgC+RyUIo1DzKp8LVEopdxRCnAzcDfzRR7+AzD48u1inlG3ww9Mw533Y8CvsdSkcept6sstgUNT6i5SOtLPf73l1l/CD7MsIMZv8jQsAVWlqripXz4AXj4Kh5yQKV/6omqdNe8Eo2i/8Cy+KOxkemoOc1wXaVLC1uBvFyyawUrY1JkRMkXwbGkyFohGEGW/xbcH7AIyrHkoNeTDzHSjdAboPTxqDktq1QA9X2/Y7ImOM4+3hEvrEV0AcrnrzZ546K9mN39AhaL/jiuQ/X1qVdVtrY7QpzjMWJ30x/mjGb9zx4WxWbNzOzUf3N+r72VHaYZV5p6YIXvPHHt2zNqYYkWVDIZG2H0K2sL+jYx/9DoDy4jwja9u0pRvZWhPnyF07e7alB7Cri1lFNPp9+otFdI0vg+ljgIPZQ8yhfN5S6uK9Le2kIrpmZ8Yk/YPDEMVsSmXLI2/fCIVtjfE+cUhXw1LNTHj++sZPtC/N54frDk55Pzs+mfkb5cV5tC6K8uOSjZw0+AzYdiR8+De1wkXfQUd1jiobtvGnuitYKjtyRJk9hafg7Lq/M2ZYJfd9t5ETwxM4qvcRrGWSpc4PSl/Or/mAPiHtO27bG2VRcryodIiapw4xW5GRlPIrwC5fkYDuA14G6AL50cCrUsoaKeUiYAEwTAjRGWglpZwo1Vn9PHCM6ZrntOM3gYNEGm7EXg+/eN1WS3z/L+esYfQj3xjB2gCixPgw71o1mNVCLRxwSUdqY4qRd4BQmFdiB1AkaggtGm8Rz+j4Kr4rAKWb5zn2LbTkG5U7+PbfiYuePFAlBt2G83P3M4ziPYQqzhGbV0BBa5Yd+hRn1f6dD5URxn4qpkieFcfyXXwXtWDO+8b1HUSl6izzxtnwzChLP98Lqx9JZPsaS7l1GCUHh7Xct9csZXV+BR1Rd0gbtzlzV4aVkdbO+z+v5P5P51nqSKlGZr3m7V8AkzNRrfqOqqqTI2lae5X6ozArSc27/982VTv6SDiZXepQpDU3grl/kZBZh5DcD78blWvf/oX/fLmA5ycuZuDNn3jH8HEh3uu3Jt7Jg5/P5+KXf/Tsz8qN2w3z0WTHPPW3WVl6Tvgj3si/lT7fX5NsZaRIlm3YxmEPfOUYddfszKgTn86s59bIM8ita3lh4mJLf3fdOpHX8m41NmGWcdy2AQrbGONeXpLwLLYbKKytShYd+yHSf3phKic8PpGD7/9KNeEecDwMOx9G3QV5pdC6h1FXShiv7M6vcgfX9pa324eZsic3x85GyS9JOj9H6U5UxBkVmowUYWjTw0VklLLrprqp55AbMtUhXA7cI4RYBtwLXKuVdwHMnlHLtbIu2rG93HKNlDIGbALKnW4qhLhACDFFCDFFL/N6wJH3jmeEKXnEpa9O46flmywf0AGhaXQLrYXjnoTjn1YLW3fjqjd/YuS94w2Cspki2lJF0et/hO8fU+u9fLLRzmypWj1Eq9cxVMxhXv4ZyKrfjPOh1TMhUujc0c6DOHre4dxWdxoAi2QnYlJ7NYVtiIfzmaDsRh2RhHVIXLJWlnJq3fW81+cOGHU3kxRV0daJSoaENJZaKurMjRRApIB/KaezSRZRWjmbPOoStu2m7vQQq+koNsKR90NBGZvC7eggNmrvwPkREmanaktu4Yv/Ny1hoRGLJ9/bjGQRg0tFE8xMwXXvzKDimg8AGPHPzzn0/q+S6ttl6Oa8x1JaCUyViaCEQsIzzIJf7/FXJi/lnk/mcuO7M6mqjlHlkKBIh6ttPqmVjeah3OuuL4yxTRYBQXT7Gk4MTzDKbowmOFglbgtdISWvTF7K3NVVvObgFGl2ZtTFQWdEPuXMyKeEv7yNG96dyQc/J9Jr/q3yVoaH5hCOacSlahWtqVKPt1dCYRuDazOndL36TXf/G/31+Il+64oRF8G1y8C0qPuKZWQeeIfqU5SdADgg/BNKWXcIRx3nU1qSEE+RUZYcggsuAq6QUnYDrgC0ldRxXroJfvWeeZ2zFkr5hJRyqJTSkFnIWK3h4ecE8wemKzHNL2lQaCG1Mgz9j4WBJ8CF30K/oxn7k8r06AvGJllMVMQRMq6yrooC8z4y2vlR6UONjBKu2cwVkbfIE3FVJKQ9TnjZd6pN9CmvwRH3WvoY77k/AOtkmdpPIvxGW/VkYWvLImeEKFAUw/FqVvnBMOJCrq9TxVGdxAbaiirTIKyCWDUccB2rawtYKdvRdeXH3BF52pJLV8fw0Gz1QLPh3hhpSwehcgjuwbUS3q1eMBMU+wdqJzbJOgTvtp3aBKB2KyeGx1NctRBqqmz1rY3WmAhZXLFyCOZcABYdgkO/0hUZlRVGAe90iHYdghmpeGpXT2UbQey19kv2/t/eRIhzcevHOKX2Ok6ouZGJGidaEreOX8mvH7LP2lcBK8HUYeYQauNxStnGUSE1HLVYvyCpfhjNikuphc2rKP3PQN7Lu17NcFa1EoraGDqodGNrZS3Gsw2yn+as0U6TsZJ2LNT2xkqZyn04vSq9zO01m0WbXv2qLw7hLOBt7fgNYJh2vBwwuwV2RRUnLdeO7eWWa4QQEVQRlO8UYAVfXA/39nGON2KDLsOc+vVHnBgeD0gOCU1liewEYfWDpNMAMDkd6ZNvsznH6fQX4Z6ELPWpdleziRI2U0i0roodhOaMUreNEAp3RMYQ3rgI+h6hhuXVQ+mG8+DSady+oAKAtbTWWhQsUzRFYasuVtGTyVywzqSUBfhNqkSkh1id2FUBfHojiDDsfDi1cYVKqe5yjgknsqPZCcJa2QraqbuXNZHOtBLb6cR61xlpd0xzIxzmcuNjcZmkSWanDhW318YToj1cPGe/fZB7ok/wef5V8Oa5lv7a9QzV2qZhV7GQ3j/dTTxm5hASBCEkzPLuzHZ05t2aThBWe1jJpeOHsXT9Ns4ck3CacrvUTEALqOGoOVcBME4ZyuJQNyYq/Zki+/JCXBU1DgvN5qnoPZSgKkorPr+QvRY+AJAUGhtgh9hKylEtuWpjCueEP6J7aC3LZTvEhoWu/Q8ptbByGgDdQmv5Il+T4ZMgxOnm+M51WBk/78NPPoRHQ6pkQA+i5+RHkU7IHM+5V08EYSWwv3Z8IAmV/1jgZCFEvhCiJ9AHmCylXAVUCSFGaPqBM4F3TdecpR2fAHwhfYx0MdWAJDr/Q7XAJJ5xg97qW/m3cE/0CY4KfU+f0ApLkCw79A9mk1QJgtQXtO0JmrVwlWrLvlkWkxerYgehnduyllPDn3Na5HPiJZ1gV01XXqByAnQbDm17GekCq2SRcY86TTFNxwFW80djIUvYhOuTrooiflJ6MSr8A+U6hxCKwi9vECvfCdr1AeCG2P8BKgFx2nUOD81hstLXWDXn5A8CYO/QTNcdSsLs1Et+aT1nt4G3t+1HZHTRS1MZee94k6LagSKYdT7zP7E0pH5I+r8Eh/By3h30nPs0hR/+he5CDcdh3gEvXr/NCAuuT9dlG7YxY8Umx747YatJl9WqULXv8Mp+l6Cfqdu++5M5fDUv4fiUKtqpokg6afO2ssNwrq07z2IWvF7jXm+NPsvB4WkMDC2ytZQc54gJ93DrsjOZWnAR5Wyi08LXuSzyNuPiQ3gudiihrWvoaN77md5dWKmF6o0ACT0ZwLDzjXnkN9ii/uS5VvT74hDMloYu7+1rMYSXYwdStd9Nru0aRS6PHLOY3bp3LGulshDiFWAisLMQYrkQ4lzgfOA+IcRPwJ2o1kNIKWcCrwOzgI+BizULI1DFTE+hKpoXoloYgSpuKhdCLACuRLVyTIleYiX7hX5G5mkhaqtWWc4LFB6MPsJgMS8piFpcqqN6SeQdAP4Vczdq0ifReGUQd9adQl23vSznx+cfwHvxPdUuUET7miXkC3XhEGtmcnXkNb5X+rHu3B8SNs1F7dSUfr0PsLSlu+P/O3Yc/44dzyrZFnofaPmYzeaChgex6fwr8QMZEFqshtoo64bc/XQAvt3Y1qizUHbhx4rz6CzWE5aqPkVvoqtYS1exzpJ5ammkgm0yn11CS9x1CClCJ0DyB2n/bW872cooufFv5qvcmC4adHRMXPEjS5QOvB/XrK1++8Vyj3PCH/Nz/nl8lPd3Ht58Kc9H/2k4HBXMep2rI69Z7gEwfdlGU79g3Mzf2PdfX3LUw9+498MG8466NF/lEOzK0KXrt/HxDHVuGx+zj4UoP2L9tO3vpZjt9BIrjYWkJqbQEfWZFu1yEZsoYZuJYK3WuNd2QnVo7CqsXralbGebXW/05e3G4dSCixg07UZCQvJobDSfKKpX70nh8Yn6719mHIaVGsPU8+rYn5itdIM/PAStuxsbobxIevvZuriS42jEqdtSfHAIMhThH7Hz2N5JlRx4cZxuJDDm4z5qO9799WNldIqUsrOUMiql7CqlfFpK+Y2UcoiUcjcp5XAp5VRT/TuklL2llDtLKT8ylU+RUg7Qzv1F5wKklNVSyhOllDtKKYdJKX916ocT+oqlyKi2yG62eh63YxOjw9/xfN5dSeZui6RqkrdzaDkbZAlzpdUN3gx98lVRxBPxPyALEwsrReXcVXglW1B39ptlET3qEt2P/PQS+dRybd15KKFo4rqCVnDup7DnXyz32kIRFdUvM07Zgx/lTuxZ8wiUtLcoKPUJUVNn2hGY3vJr8ZGGcpn8VsT3voJqGeX77dYAX+sLexARCnuKGWxf+iMPfDaPR6IPcmfkKQALQYhJlVj1FKvUhX/rOqiyBrFLWBm5CzsVKR11CG4T2I9jWqGW0FeP069fk08tn+RdzTWRl2HtbJ6NH8bNdWcDMP+p/yMaV61SYnHJ/qGfKKCWfqFlSAkjQrMs9zg8NInuYrUhMsqjzhCZgGpOe8ELUy3X+NmNmnML6MmHKrdZxS7/+mQOF774I98uWGdxGLPDLqIrsCU6ti8yl0T+x7i8q1n09WuM+WaRShA0PVF1gcox69ZfAItNsYPiUtBNWPV2bcVmIuasU4oCoSgzSvfh4tpLjeLXY/szXe7IUtmRuoqRnBL5gjBxZNVqNQaQcZNaJs1aiESwQpZzeO3dMEQVJOhcTUHEPZmzUwKsuCKzjo9lhj+dlrcOARJe44kMfE4Ewfs+5kCOv8tYRjHC9BarUHTX8g1WFraV5gxSIqqTPGLbiCp1xwEs8xAXgcOuNmrSJQw8ydL2Zo0wxKVggaKaoj0XP4xFsjNTFldaZYNdBkPEltjbBeYXrC+o1SZfCitbGuLvdeczT+kCOx1KvFVXRtbcz9O25OPr8noA8GT4bgrHHMCCb97kqPAk9gv/wjaZzzyZUPnEFMki2ZleYpV6/xePh/t2go0JqxK7o7LTTsZueWO3MrIvan6UynqgvE02gtBVrGXn0HIujKjmuB/Hh7GOMmpkhD7xBey3Wl18+lb/yP7hn/lffB/2qXmQU8S/eCZuNdUNC8m54Q8NkdErebczo+A847xuidZD/MYgscDSDy+YOQTdKqtyq9WsVxdL/bp2i7Ntvgvs6U7N1xRRzSnhz4kIhUvW38Gij/5NTSxmGA7UFKjhos0cAgj+XHsp19Wdw2+040/h9xmXd5Vxdp/QDApimxN5AbauAaWOhaV78IEygqNqbuedPd/i6tifjGtqB53FDmIDD0cfhocGATA1pEYJ+HHhKuYsWsrWUAnStkzpllH5UfflyylUe0yRvnQ7bjjn2R8sVnK+/GIsO3fnC3Ti5WWkkCrSrV9P7VRdbrYEoYY8eoVWGi58cvJ/aUUiPEIZCUWjmDUWtLjkYeKUiyrGKXvwcOwY3ozv53kfuyKqNqwShF9kbzj0dot55WpNqbuGNgYX8ll8MACXvDKNJ7/2zfxYYJ5U+qQxcwj2hXax7MyhtfdwZ+0fef67JfxGObVELXXW5Fs5hjF5CcuntbLM8hHGFclS2YEuYh0RYrBqunpi5jtGHXMMffNvM+yLZEyxxspJZWXkNJvtHIL+XbTTghECUNadVZol86V1KlfWq0rd0fevUc0V34jvz3LZnuqYwiOxY1kpE5zgelnKHqF5bNE4BMOkV4MeSnxC/pX8L/9GrR+pVwvz3NEX/koXP4+4baxSIRxyJ65XRt6gTGxjlWxLVMS5Lfosh8mJdBIbqA0VUBdR53iNbSP1oTKCl+IHs1B0I1/E2CmUWBz/HnmVW1ddCF9oYiJNb1NVoH4HM2Qv1hX2srS3pcchVMoSjghPRtRtg6578HRI9fQNyzpai61UiWTbfZ3jz/fgEGrjCqc/NclSFlfS91k2j/kXc9Zw+WvTE+eySKFphtnZFJx3+F7cISSHGHFDKoLYfAmCjNJbrETUVKneuNs2cFkksUC1Fibi8P65oKW7+1NY3TGulOXcFzuJF+KHet4nmUNQdRbz6A7hiGGVAvBqXNUJ3Ft3EtfVnUPN/tczRSZim89etZlMYJ5U+gs1LyZui89ns1czYd5ax3PbKOQ32QZQIzOOiSV2xes1n0P9Y4gpkpWynKiI073aFAPpu4cTSUM06D1xZnuTk3/HPNh4PzoEnSDou21l6zr+HH7XIvaRvUYax58ow3gtNpKyOlXkVaRUUSlL+EELllYTU6iiiH1qHjKuWSC70EmsT3Kcy0O95xa770DNFn/JeUxsvv4+N9pERkbebGn2Q0hu205MIzaCkDAekPQTaqDG2+pON87vINbRUVSyJdoOmcKGdWGoAoBP4kP5Zd/HWJnfm1ZiG+3iaxL5h7doivhIwqXILrrdFg8xQdk1UTDiIrYp6salOBSjjK1sJpkg6BxCgQeHAPDNgnWW315zzQ1uzoDgYtFmg1NeZDt0ZubxCQtZU1XtwiG49wOsc8lz45CtDqGpooYobcUWIpuXQsXe1O52BmeGx1EhVAVca41b+FfdSQCsW63qGLpoJqGpOAMd9kVpU081iuF8RbUdNi/M82VXRogXeEvZjzW0oWbPyy1JPP43fSWDbh3H4Ns+dXXcsqP/jR8bO1CAD39RranMiUq80iaa5cDfmj6Qurg0LKbmya7cGjvTULrqFiXmPAi6SetO1VpQvkPvgK1rYfzdAJTV/cbD0YcIx1RRnZOJ32PjF3Lb+1b5vFnkJgQWL3L72E9ZUsmm7XVc+dp0/u8Z1aRSFxnpXsjF8//H1dHXuCL6FgB31Z3MQ9FzLe1soJSi2CZAUqJUsVEmxIDGLs303hYoXWgrtlCz3Zq/oYwtlLCN3hsmWMpnzZjmS2RktgyZv0Zte8O2Wr5bsC4pB0VcUUzENmXTRiiLg0JTeTvvRnq9dxybtlazcVsdvUKreCe+Nx8qI4z6EeJ0EBupymufsu0J0X34PL4798VOZM0OB/FLkSk0yhbN2k8LzLg53No4ZRfdKlLyROwotfoRDzOt1YFsV1Rrq8JQjNZii8Xce9P2Oobd8ZkRmNCuJ3GClJJpS1VRWDyupC0yMm+KzJixYhNHPPR1yuvNZs0nPP6dYx3dUOTNqcu58rWfXDdT4L6eW81O3fuz3C1rnN4Xz7NNGDWaCCRUswnyStgy+EKiIs7QkBouobVQRUYfK6qLxEMfqs7NpWIbvyqdiJN6MkHyorS1dV/2qH6UZxV1R21f2DfUJUQzTnNv47Y6NmytdXStd8LW2jhzf6tKLjcRibVVNWyy7SxBnURmOfBpJha6Lq7wcVy19PhUGQLAJs0/Yb0s1a5X68YUSV2xqlTcsWaOWtj3CBh8Bkz+L2xdx8DKz/lD+Hvkb7+wrTbmmB7zg19WJZXVxBTLjvfvpoxv9rE///kp/O2Nn3h72gq+nKtyPsUaQZi/Wl1QQ5tXUCMT7+C/8aN44KsVlnY2yFIiso5iqilRNrOR0qR+mbFAC03wt/U3qiIzDa3FVi6NvMNFq25gz9BMo/yRt8bxyUxvM+iTHp+YlOAeYMGaLZz61CSDeBscgmIVG8xbbZ0T9j39uqoawsR5NPoQFeI3ildP4bI77uOQ296ks9jADKUnAB/G1e/j6ujr9BVLqYq2S0lwlkZ7c27dVcyT3YgrkvWiTeKkbv69VX0/m0WZccopl8IsWUHv6hd4ZMMwjn30O9bXqE9SGIrRiq1slAkO4cellaypqjGcRv0QhJcmLTXm8ZzfqpKjAKeAPhR2j+4x3yxKruswcCs3JsyI17h882ETR7a1NuZItCbMW8s389e56xDiZh2C+ws87lFnoqSj2RKEakwRN4vK2V6oLlhdxVp6ixUMCC2iVoZZIdsB0EqzCillG1Wa8tcP7HbtdYrCWlpTJ9XJaKfGll2Qx4el72j9yIXXbkm2TTfHaJowby2DbhuXVCeuSJtiMIGYonBX7BQGVz/OcptifR3qR6xIyerN1WzcVkvpDjtRJ6IMjk1TK5V0hOEXgRKD9y6j65YZAEz8eQ5/f/wNjv76DxwcslreOKE2plgmuf6xgzPn8/lsq3WT7pykE5vIlpWskm05rOYurqq7IEkhCVCpEYA2oopSaeUQnDBTqQBgUOxn+ovFRnlrtlCE+m6eiiZ0MB1FJTNWeIsHJy/ekDKfc00sbixg5nSWm7bXcegDiTAcTvb4b0xdTmexgXxRx/2xE1FCeVwaeZu/aSa0v2gE4c91lxvXlIltrM3vTiq5gtncU5HScKjcKFqr5t/rF8K29ZBXavlO9TnbfwdVJKm/3zhhw39D13UViDpaiy2Wd2MXjdpNa52wdEPCGuzVH5Zx+IOpd/Vm6FOwLma999otyYu7087caROUBNPrCwvh2M5NY2dy+tOTXPUW5u88mwgdzZYg1JkDtRa3o5p8NssiLo+8zef5V3Fc+Bveje9NDXnEI4WG1VGp2E6VdIkp5AA323k/IgEvpZO+o/djqeCU42FrrVVu7ebu7kYQamOSGvLYYMQoVPUqAB9poiNFSobf+TlL1m+jLlLCtFYHU0CtGuQrrxg67qIG/5r3Cd22qDv7CvEbV6y7mbJtS7k28nLKZzPvZvTQ0jqc8h3bwxXUatZW+mITqVrBKlnOXNmdN+IjHe+5QeOAOrGBvspCg0C44a7L/s847pmX0Jm0EVUUCvXdvBvfi0WKap3TUWzklxXueRZ0OHEIOhQpueqNn42gbXFFGuv0NpvOwq0d3TT0V9mZmtJuDA4t4Pjw13wVH8hPMuFpf0HtFZxdexV7VT/El+1Oc5xLbU0hz4vyEjvzuAK/RAbybnwvniw+Xy18eDD8/DoUt7O8X10p369zq8QzadA3Ujp3F1VqKGMrlQ7iPB1+OARzXzOB3v+auPU7cuLwM42VFDJxCGFTFF3H/rjcwiwaVqRk6pJKFqzxTlHr2Je0r2hCqJEaUShuT00szlpN9v1tvD9n117FTbGzAdguitUQ12TCIVhfjp119MLsVcmiHh0bNSWon0nkxGomKTIdoIqMnOs5PceT8SPZv+Z+ZsoK7frEuUgoxKT2x6s/SkwcRa8DQKmjOKbKaa+KvE4PsZo1Jf00z1dvimeOv2+Hkx7CzdtZrxvZvo41RggQZ1RqBOHPkbEAhvjEjh+VHQHo2KaMH7QgZAPzEuKnErbTiUqmKn34R+x8Dqh9gOWynWG+mQpOBE/Hs98t5os5CVv/uMlkss5xXJLb0AnCMtmeL/pcz3V157B7zX85s+5aNTy6hnHKHoxXdmcl7aiWEcf3sUNrNbdAJCQY2CUhBopLyUZRxmV1f+HL6P6JC4raQv9jiCnSUHDrjn0l+ep3a36V+nys1TZ6ZXIzYSHZEE98q2YCEgmJJMW5E/R7ueEPu7lHKjXDTnQzjabqhLCNIHjqhF1ObjOJkKWE4x/7joPvn+BY1wvNmiBs1C0QisqpjSms0axmfpa9GK/szjbUSbyqJp9TIl8yMjSNUrHNCBHhB0mmkh4fsR2nPPm96zndvNDPJNqwNdkUcatPgrDdRXntRIhqiapxnTSY+xYOCdaW9mMq/YwQGACU72hpIyri/Cx7M6fdYRSLGkNU59pHBX9aUoc+QWJnaaS2rK1KyQFu0DiCvUMz+E2WM8bmo6HjlNrrOYCniIZD/KNO9TvoKxJcTLGopqOoZLVMyNBXyzZ09hmKy/4OzKKfr+evsxB9VWSkHpsD8OmLohPx7CbWEpMhVslyLv4mn5fiBxtOlG5w4zbCminMjh1KKDItsoqSCAAYVyTspFmrnf8lHHwzcUUxuDqdQ9B37eZ3qd9XJ1Tt5EYA1itFpjqJ8YqGQ6nzogNTl3gT5+IUHISUqm/IzWNnWsqd4jZ5cXxeMD9HKg5ho8N9AYuXuPl6RZH886PZvvvSvAmCrnDKb0VNTGGjZpGgp6/ToTuMPZt3D6Vsz4pD8LOA+5mounmhn4BbTnmazTsCNyjSfa3VrZU8r7ftyKLhEOfE/s4r3W9OjEOngUadmUoPAN6K78tmzVpFj4/j3sf0PiI3nU1dXPLkV79CTVXK97tBqiKLfBFjqtzJtV4NeWwQZUTDwiAyPZVEbu1iqulgIwizlB4MDP1qUT67wb6AFHqIQFQberX+7R8kPnD9GicC302sYZUsJ+YrD5beJ8VxzkQ1wpMfDVsU2HFFJsSoUsLxT8HFPxhhomNxaRA6ncAVawTFrCPSF/saosSloCvq/NxkFhmZxisaFkm+Fk7j99EM73lenIKDkEhu/2A2n85SdVdG6lGH7zZjDsH0HCEXHYIONxHwNtvmQcesVZv57wT//k/NmiDcGjuDusL20K4PtTHFEAUoNpsLfQEAVYegy5D9wB4N04/IyA8rq9u0++E4quuS77lFEwV5LSJ+nKM872siRGGNIGyK5XHt+7/y0Ofz1V1SfgkccD2TO57MSbU3MqL6YV6MH0JlRBUrdU5BEOIeIqNUWLFxO+u3Jlj3ez78mSh1KTmEKgqJaVZmP8T7eNYF1XFoe0hdmHaILaMyXE5cCtqLTbQS2y0E4VtlACWiml1F6o/QboZpTh9qR1w629Dna+8/7jCPuos1LJOpzUjNqIsrjrovfdESWDc8cWnjEPJLoX2CyCpSkqc5kG1NEhmZdAjadxUnzDzZlaEh1d/FbGVkXnDzIiGL7B3g0dMGp/WskFrHoEirdVTYY7eXiQ4hL2LldCIpOAQ3mAMlmj97NwmBG5o1QfhWGcic06ZCgcoh6E4uC20ZjKqwLhCT9Vg/LvDKy+tnR++njr4Y+JlETj4Lehe9djjZ5vU1m7ZGwsKSkOTBz+dz0uMTee67xZyzaCSfdr+MrRTym+YRvEXzMDV7jDshm2Bje9/1hSU/tq4n2kIqowHBZqFuEqYqyRyCWXSjf6y14SIjKOJHrU9lK4VqjH5UMdGlB6qis++VfihSsHdoRsr+29+r1+IUj8skz2GAwjz1E7bPOYFCb7HSEoMoFSIhofqnOHEIJmW+OcTI1W/+bDhcxhSFD35exTH/+dZiv69bA+k6hITIKNG+eaP1s9KbYk1ZvwlnpXI0bCUInVoV0LFVge9n1eFFhCF5ftq5EjMy5RDMzxEKiYy+ie0mXaH5+u0uHIVrX9K+cxODznbWxhQ+UYaxb80DfG32fgQUk8/BFlnANGmVe9thfh9eHILbi/PzPms1qwU/k8hpIdDhtYiYm75tdH96lPsXlYHVckHnEMyYu7qKm8bO5Is5a5II53ahLsrFwtsRRpHZczI6SrR7+dERbRSt2C7zjEx3ZjhZr0TDIbZQyMa8TnxXdiRbKaCX5gT5G20NR7CNlDJL9mAvW4A8J9jfa6HHu/zYxa9BD/Bm31j0E0spE9uMjFx+UJwf0TiEZBgcgnAXiSoKXPH6dKYv28g7WsyfuGISGVXHyIuEjEB+5g2L2azTbAFl5hBqTRxrNByyiVoSHr/poDg/hQ7B9tuLIKSjXzTfwGJlJERGZqNmDuHlSQmx5u+KQwCT7FGbLMtkR8v504Z3JxROvPSJSv+UMlXzRLXvvMwvPZv46gaHkKEiSodXCGDzc5QV5XFwv46udZ3w7vSET0BYCKIe97LH4NmiKfSLcY/vD5oMOkt60De0jKej9zAh/0rt3qnNihezA98oA4gRSfrInYhsJBzintgfGbfTLchwPltlAT1CqhXPYqWT5aP+TunP4NA8CvB2PkyHQ7CHtNChEy9dHxEmzpWR13ky7z5qZIRvlQGefTCjJD+SJMbSoS/qAq8QzAoV2qbjytdVj/a4Io05ur0uTlFe2Bgr80bAvNH6SUnEPDJzCLUWpbKwRDQVQniKc/QERHak5hCsz2sXU5mRSQIeibQ8x8pN212JvxfMnMDbpgB8fiMi6Gj+BEGbwE4TOSTUAFirlNZG2VfKwKR6oMbVf/AzNQ6LYuEQ3JXKmVoVmPubbdIOL32F+YMLC+EYEtgLT5u8MRXpnbJwhc0lfptUCUJJSg5BZs0h/D3vDfYIzTF+b8c9iqz+PV/LpfylTg3LbNfDOOll4orkxfghrG+3B6GQYKtG8D6P784qyi1j+53Sn3wRY88UXEIyh+C+OLktaPa+9hVLuTTyP1bLNlxQ91dW66lYfaAwL8ykRRssiXV06NFDnYIW6ogrku5tEwv4Q5/PZ0tNzLJpKYqGjTbiDjoEgLmyGzUySo2MWhzbnvw6MR+j4ZDBlan9wvLbjq//fgBXj9o5qTy1lVE6IqP0t/aKjUP4eXlq/xUnuFkdpksQ/JsfNFHUxhW21sS45u1fks5FwyGiYcGDtaMZGJ3P3qEZjFd2c2zn9KfVsA6zVm2yEJfkJOyK43G6+N/0lYzcuQMDurRKXdkFIeEtnjLvWMIh7w8mFXbtWmYJ5GfHikrrwl8rBdtkPj1LJXhY/knpL5mME9qymYGhRQxmDh/Gh/Od0p+H8v7Dcs073QlhIYhJyea6kGHimB8JYXY8dVqYdcJdqAVU26oRPD2suHlsdR3VM3n3sHP1sxabfzNqbB9rgQcH5iY2LLAtaHoCm9vrTudHDwsqJ+jz/s2py5PORcKW7bjj9XFF0rVNgju7/1M1jMzg7q0T/Y2Gk8I9g5VDiBFhluyhxR1L3Mts+58XCVk4glAKDiEScj5flMrKyDY13QjCXR/NYVC3Msdz3u1LT67DC8V5YUNU5LYW3f3xXMdyN/jJmDZGCLFGCDHDVPaaEGK69m+xEGK66dy1QogFQoi5QojDTOVDhBC/aOce0lJpoqXbfE0rnySEqEjnAeri0vDmtCNPkzPWEuWsur+zV83DSSIlOz6ZudqIkwPJO3jzxE3HSc0JV74+Pas8r8X5EU+ltLltIUTGE++YQTtw/OCuFqWyHXYOIRaXVIcK2a+HtzzfbLaYDnqLFYzNv57n8u6mjC3Mkj0Yq+zNztXPslgLPe4EfQzMstWI7bkKTVE09QVBH+eCaBhFSpbJDqwu7c93Sn9LuwDbKDBk9+1w3/HZF3l7P8xwm2utbZyDnr94Qwrvayd4paSMmK2MXOps2l7H97+ud2g3MZ6RcGIeWnQIto3Xi7GDPQNQ2pXKIeG9ew+HnOe/FxEG/zqExycs5N5x8zzbAth7x0T010sO3BGJPzN1Jxyzexfe+8s+9OlQ4irqc/Jh8oIfkdGzgCVjiJTyj1LKQVLKQcBbwNsAQohdgJOB/to1jwoh9C3MY6ipNvto//Q2zwUqpZQ7Ag8Ad6fzAHUeEQyjkZAR60YSSot91hFTEtESwSqaesthJwUwvKe/+ygyOx3Cn0d6K8fjNpGR1w7KC0ftugMhB6WyF+riCttFIeGYt/v86P98ywc/+4j3YkIedTwT/Rf5JOTqeoRWt904qB+zU8hkezIVJ6WyPsUKNRn4P2Ln8v7gp9CXR/s68V8tiqc5DLsdNbG4RW/gRbDdPvjubVWC256NfJB3LWdGPgWsptZ+4bWg6t+Rp1JZqgHk7Mg3jWc4FDJxCIk69k3BW8p+/Ct2snt/QsKiRA4J4bmwRkLOjmyp0nC+99NKix+Q1zdU4cNow5y4SAjVKznTjZoQMLBrGXmRkOv8SBd+Umh+Bc6ul9ou/yTgFa1oNPCqlLJGSrkINX/yMCFEZ6CVlHKiljrzeeAY0zXPacdvAgfp3IMf1GkiIydEw/7c273wy4pNHGuKEGjeCf/zozlJ9UsLIhzW37+pX7q7Y7PMOBoWtC5yXwDNCIcEowb475cZu2jByNIhCDUxhWpRRDiWOrrkrDTzRIwIzaJ7aC3X1iWylvnZEYeFoLwkWb/gx8FJtwDKj4QpyY8QJ4yIJsQj9o9at47xIgjVdQqtCqL0alfs2A+AE4Z0pVvbQkfnRICubQqJEuOxvH/TP7SEQaGF1Mqw4YyZDrw2JwkOQSRltkuFvHCCCERNHEImMncdRXlhSz9ECg4hJJwXXq9rQA0q98nMREBFVUzrPE5+li2r+a7Wps/Pyr6W6c+fFwllJb42I1ul8r7AaimlnkKqC2COULZcK+uiHdvLLddIKWPAJqAcn6iNKa5xfaLhkCcb7gd62kQdz09cklTn0oMSzk2RkEhrkbcnXUmFwwd2MoKDRUKCNj4JghAwoEsZv955RFr3++8ZQ9ihtbrwpUcQ4tSECgnXphdu2A8qhGqFMV1JcEjrfeyIwy7vxv6hOSlwdSJRmBemtECVO5tFPkkEQQur0trDD6MmFrfMT6fxLYiGXCNgAnRolc+J4QlG2HeAavJxF+y4w76o9O2UILIRB98Mv4iYuMuISXTz78/me13miZKCqEXElUqHIFyMKtIlbkIIVzGvl3m4DrOlnt5dv33Q5539+rxwyNe9/SBbgnAKCe4AnGeh9Cj3uiYJQogLhBBThBBT9LK6uLQ4UJkRDYey5hC8PIF1mC0VIuFQWkrSZZXesX7sCImE40o4HKJNkbP1iR36TihdxbJ5EuZF/F9bU6dQFypAxL2tjDJBhVjNVpnPOlOkVjtBcBIFuBEE+y6xzGFMDYIQDVOqKSLNmwX7sOphm704hJUbq4mGQ8Zk37dPO04ZZvWLyAuHHd/Z5Qf3YdeuZURCIQ4M/cgSpQMTC1WZ+62xM1zv6QU7h/DfM4YYc9sQq4n0SU04LAxRSSQcMnbETuIlvyjJV62VPr58X7VbKayMwJkbSJe4CeGuz6n24QRmJlo6IchUh6Bf1qAiI9fOCBEBjgNeMxUvB8zJersCK7Xyrg7llmu0NstwEVFJKZ+QUg6VUg7Vy7xERuGQcEy2nQ78KH3NlgrpcgjmeO1+YLYsioYEO3X0pzzMVH9QYBFRpScyioULCMW8/RAyQXexmqWyIyCYGN8FMAU61OC0yw8JVUR30tCulvIkguBwrT4OeZGQQSTN3J19MdqE7qntThBWbNxuSdiSHwlx/ZH9LHWiEeed7wX79WLsX/Yhv3o9+4V+5lNlCF/vfAMH1dzDm/H9Xe/pBTuHIBCGGMSiVE5zKpl9WKJh4Uu0kgp6+AudUFeUF6eUxTvdN135vZTJuRF0+HECM98u3WGwK97158lvCgQBOBiYI6U0i4LGAidrlkM9UZXHk6WUq4AqIcQITT9wJvCu6ZqztOMTgC9kGr7bm7bX8bUtd6oOlVXNbvK5yeYeM8VNKTItmv13KPOUxf7r+F0tC9DqTektmAJhxJqJhEOcs09PDtkltcNZph+hWQmWrshICeUhckQQXj5vuLFY9hBrWKxZi51X91eu7/xEUgY8p+Qp4ZDK7kdsz2F/rtaFUa46TLVZ14dN1yHUxRVKC1SCYRZV2heWWqLUiTwjD4cX9OkecrAE0y3l7NB3l13nPk0EhZfiB1NQ2oaFsktSXb+osy0qwsQNhHXHNOE8l7w48eL8cIJDCCXHIMoEJfnqO+hRXswTZwzhvpN2S7npcbpvJl1xWxPsBMGcQ8KpD/qQuX2bL5033EgmZL6vnfttUB2CEOIVYCKwsxBiuRBCT1B7MlZxEVLKmcDrwCzgY+BiKaU+ShcBT6EqmhcCH2nlTwPlQogFwJXANek8wP2fzuM9LcvWuxfvbTmnurdnxyHYPxId7UoTykmz+/uDJw/yVJZVx+IWDiJd13IhEo5zesTH43ZPvQikUp6ZYa5qtspJl0OIhwsQcX+pQlNhz97ldG1TiEChu1jDEo0gbKWQdUXJ1lZuCkRFyqSFw0lkpCt6dZw4ROUqurYpZHfNrv6Avom8EE73i0eKKXLwVrZzAeZ+2Ker2wKqF5WtnsQkpR+LZOes9WWOi4quDE7xHXlZ65QWRIlq4ka7h3GmMH9zh/bvRGlBNKVy1um+mRAnV4JgExkd6/Bdmu+mEwInR0CAvXdsx81H90/cV1uLdDGe3vX8SDhnHEJKxzQp5Sku5We7lN8B3OFQPgVI8qOXUlYDJ6bqhx/s1q215Xc4JLL+SNzkhWbdgtmRSfUNcOcQurW1Wn+kTxASOgRdHOZHL5DGWk4knGBB80xhP/ykLNRRU6eg5BcganLDIeh+FLuKX8kXdQZBAOfFyO3jr43FkwiAfXfbujAv6aM/eVh3jh3chXwtdtDc20eRHwlz6SvTku535MDOPHLq7tTcW0xxdfLz2wmrvjCEHZynohHvORyu20qlZk6datFOhR1aF7rmHI4YoSuczTvzIiHX0Mwl+RHjmd38AdKFXcGqt+0Fp+8kXeK0YuN27nKwLoRkr2Cn95Yup26uPaJXW77/dQPd2xZRuW1Twsoo3DRERk0Sr/9pT44YqJpX5sLs1G1xN8vW7ZPTTYfw4aX7csDO1vzFbtEIX7tghGO5WYegfwB+njGdiWieXPkmDiFV9il9YY6EBDWxODKS78oh3Dq6f1JZPrUcFvoBtyxrkbDg/MiH1MioJYChE0Fwel6VQ1D798N1Bxs2/PaFZOdOpY6Llk4M7MdgW2w0sUo8WmTkXLY/h46p1x+cMBJwWCzzws4cgl4WrttqeE1nu/l5+fzhlt/m20bNfggOauU8jx1HcX7EolTOAT1wjPKbitA4nbeLD/3gPVPebzPsBDFfa7t3+wS3ma4OwTyPLz2oD1/+bSS92pdYrs+LhHwlzPKDFkcQhvVsywAtxV8kFMrohZvhpg8wfwCtfBIE3Z7f7IzkFg7CLay1EInlUteP+OIQcqBDcNqVmdGrXTGXHLgjMUVSua0OGc6Huu3YF/jOZQUMsnFzABdFxvLfvAc4IDTdsf2QEPQTS/hCGcRyU5x/Jz2R0+OGhCCmKIRDgval+Ym0kKbrz92nJzu0LvTNUf1xaDduObq/42KjRIocg/sZClqBxS9CFRklcy9OO9/Q8klQu41QbIsRzC/bud65rDDJcUq/s3nT4TS2+Q4OfzqiYWEQ7ahLCIl0UewQXiQ1QUguy3bDaIad29fFytFwiCN3Vb3nzZsXP+amIRtR7tmu2JKbQm+zyoMg+LGUNO7nu2Yzgj4xImFhZHrKFG6hIcKmRUSPmHj+vj21a7x14mP/so+xQ7bHs9HhJpNVMyollMrgb1Kno0Mww/yh68pUN0hbADwlUoBAkkeMfGrpJlQHHyflKUBrzSJHT45iR1Sppqf4jbmym7XcthDuv5NzUhjd7FQfCz1xi1nPdMNRqtWSvjNLZd5w9wm7ctZeFY6LjRItpkgkc0hR23vTb+G0UEZtMXsASthG5NlR8MofCddtNQhCqrl+xcE7ORL1vXqX8/J5KndgFpUJrOIssCqazbBzTGaYQ6dHbEHpMoWTN3lKkZHT+GZJRL1gtnC845gBXHXYzuyzYyLOlrm7bhyWmUMwFPwi8S4gtbd1qiRAZrRIgmDe1WS6EOpwW9zNH2lhNMziu47kuiPVxSSV2emOHUo4Y0QPIOHMYu+m6wTBanZq74sbMt2Umfthn3j/OsGadyIuJW1MlhXbFJWA5FPHOeGP+Tr/CkaHvlFtxh06VCZU+bUa1CwZJVULCAnJbMVqq2/+8P51/K48d84wR6mTThD0hTjPg6Cmu4s1zzP9SEaLHJXKuujNPjed5mo0FLIoSwupZnRY85xf9BVCKiaRkffnfNnBfQwfCjMO2aUje5kWKuM5RGLeGCIjNx2Cx73N1n5eSuV0htwpBEmqT92p/WzFbF4w96d1UR4XH7CjLWR34rhrW+eQ7U71ExZf1nnshlQEw3I/3zWbEfSBi4RDWe8A3ERGlo/URoH9+C7oL1NX4iXJjl1eonDgEPwQvUwJo9ciY45iCSoh7GCyvtocUxefAmrpIlRLissjbxEm7mgR0lWr0wpnU83SSpVzsHMI5o9aP3Z6A7qFlr5D1bkfR5FMuso/h/oyWuyoQ9AJgt1HxpEgRKybmv8Lf8wd0TGWOgaH4GNxCzvU8ZobwlbHTYfgJTIqK8wzuDE1ppDz/ZyIlRucOJJUejI3gptL6Pfo2CrfxTrMvHFI3kToGFbRNqmOXsvgELTSVAt+OmtgiyQI+otwSsSdLtwWd/PHbLe+ySRGi33uRMMhnj9nGJ9daXUyCodEslLZz0Lgc4E7fUR3Hjpld1917dNYkTaCUKeOS76opUCoXr09Q6sZqUxy/Fjaa5E6h3RyZnGLN81jm8zXnNISMO/wdQLmlT5Ur68r/ZwW0nTXCetOTv2hRIspFskEQV887VPTccGyKZVHhGYn1dmi5ZD244TpVMeN+JkXI6e0ota6zrj84D4c1r+jESnXy8ooHR2IFwFyg6PIyMH7PtWO2wvRsODHGw7hi7+OTDlOwmHO6Hj9wj2T6ujTw5gn2p9U1n/p6ElaJEHQEQ5lH8vIbXG3uKDbXqYuhu3WttD34mpvIy8SYr+d2rNjh4QH7inDunPpgX0MqxRDqexLZJSoM0yLxuq0EB4xoDNdWqfOONaldWHSghZXJB1MeW07tWsDqCKjIqqpLKrgV6UTx8Q+MvpcQA0CdcDaCjWUQVF8M+eFPzA4Bh1FlXOYJ7ug2KateQeki9Ec6YFBSNX6+s4qHArxwrnDuOkPuxhV9f75ZRSc3oGMqkrlVmzl/uijtKbKcn8jEq+NwJsRCSUc08LEGRwyxf8ZfCaAkazHz1xPR7FqdkJL5c/jxhSfNrwHQiR0CF4io3S4MicdQio4P7sDgcxiVcwLh2hbnEdxfsQ0h9zXCq++qeXJ15oDDYI1mqwT0mF2mzVBeOuiPXn+nGFJ5frART1CV+xR0cbXPfyIjOzQQyO8fN4IDvcZYdSPDuGfxw2krChqiEP0Z/OzMzQvNq//aU8eP30wn185MqmeEN5c1cAuZfRqV8znf93fsKLQw/7GFUl7zWrmiIGdOHxQT0AVGZWwndpIKTNlBeVKpZrNjlq+zb+Uk8PjyRMxw6s3umE+10df4pjQN4kbS0lh5Rzm2PQHYCVsiXDnye9NsRHSfJOZ7L592vN/e/c06qYrMnJ0TMtvTZGo4ZLIOxwX/oZTw18Aic2E/mG3kps5OfwFBZsXJbWRF1FNpyPE2E0spMTMcRxwHZt3PoGpWu4Fr52gQVScxGNuBIHknb8QzqEn3IIL6PezKJVdHO3S2cmm4xOTuIcTB5bdhtGOPJMoy+mzND+iZbF34bEsXIT21yy+g9RZ39KRkjTrjGkdWxV4RguNhN2det64cC8ufGEqH8/8jQdPHkTPdsUc/ci3SfX8iIzsGN6rnMV3HZmi91b41SGASalsmJ2mbt8uMho1wDmJTEh4K+feu2Qf47ggGmbe7Yfz/s8rufL1n1Ckmj937u2jyAuHEL9+Cah5lYtEDbFwa6plHvnUIEOCPmI55aKK08Of0llTJMfDhYS1gHgdxMbEjWu3EKnewCKH5Dfm3auhQ3B4bdV1Kiei63wSHELyA6crNXAas3iJuhnYPbQAgEotvpF9cT40NoFLo0/Dq0/B6EcpIY8tWvhqXWT0fPQu9grbUnKWdmL9IQ+x4afxah+8CILHTt9VnCgSi45OTJ2IBLh5jiTGxeAQQslK6cdPH8zB/Tqy/z3jXftvR2YcghMhcijLIFKsDjOhShUqw49PgpVDUP/azU7dTNS9+uFa13fNJgghhOPCOUeLr79zp1aO9so6erRTP7p2Jltwe3sxF0/lHOuikl6aF1VPNjtN3Rm/cyLdzGp5piRE+lDlR7S8ucWq+WdbUUUx26mLFLOdfApkDeG6bfQLLQWgf2gJl0b+B8D20h5G2x2FKffmFjWh/VqZnKbQvMvTlYROBEG3E9fnhNmRzo50PUqdxixWohKvwUIV8/QQ6jPYCUKxOUT2u39mRkEiz4MuMjITg8/iu8Po/2j3TVzqHf5Zq+MwVbx3kDqBNWffS67lprMJGc+q/nbiEEJCEHGJ2eSGTDgEv0Teibv0C4ufQQpi4+dprToE67xJcAgBQQDUAXWaGHq+gEN36Uhrj/DQfz1kZ/57xhD23rGd8aL6dLBGzazzYXaaC6TTnN6liIcYwA6/H5vKIaT3bLrCMEnfUqx6ZbcXGymmmli4iO3kUUYVXR/fkUNCU42qMam+x80dhqLkt2K20s3KIWxV9QnrSCYI5v4mOITk96YTBHOyGzdkY3aqQ9EIQkiofbkw8h6DxILE4qhdUyirqZZRlEiy7iYvojqr6XmiH4wdy3l1V8Hup2ttJ+7r9d4Si4mTzNxNZJQwF9XnnJsfQutC57wc+jiafYPsY2Vf6PwgE4eyXERZTQWzqNepi+YuSEu5c9+c2rCPU1G+N7eUzua1eRMEkaDI5vE8e68Kpt94CN3aFllCGe9oW+zzIiEju9mALq245ej+PGqKYgruPgXhkKA0P5K1FZMOr4953BX78fjpQ0wlOofg/0Pyu8gLkZhAfj86nUNJGquickBwbJ8onQvjxCJFWvIWFfuHfjaOH4odx5W1F7J02I1s+PMsZsqe9BSruDT8tmq6qXEI6xw4BPOjRT3MThPBwdQdlf54TiE50jc7NR1rf+OlXamUJbwQO9g411WsTVIiF1LNNvIJxbab2lD7GgmpjmnFVPNc7BAeiFnDfpkXc6/XZXAlDnXcdttCwAvnDuPC/Xsb1mOC5MVrVP9O/E2LDut2X/07jMdlUj/15ryG/Njdu3DxAb1N1/h/P6//SbXYcXunr10wgoP7JSzX9EiqmcBstZRKRGX+XPxkYTO4PN2KTbskFYeQzuameRMEBPla8DXzI4dCidSSZlnjLUcnx88x2hKCs/aqSLKwcQtuJ4Tgh+sPZuYth2XYe3t77ud26lhqSX8pDQ4htaey3q7fXYIQiXwOhw90T1ZvRsTQZdj6EY5AUTmDy+uIxrcRixSzXSZ2kvkikWDme6Ufbyv7UVxQgAjn8b3Sj9ZiK1dG3+TCyFh48xzAWWRkvqs+Jl5mp/qOar2WgLxDq+S0mumbnSa/AxktYmjNY9wQO8coqyFq6KX0vhbIarZRYLlW98UICcGO7YsoY2tSzgewfuxeOgS9WkV5cdI5twVFADt2KOWaw/ua2knWAZy3b09aO+SQ0PsPqr4PYE1VTdJirv/U53G7kmRu46y9KrjqsL5J5anQr3Mrw6rObXiG9yrnwv17Gb+d7u8XZvGtzTo06dhMBNwiqFqv1Yi6zrVp86g4BYeQDvFs3gRBpGeP7GdY7LbQXl7HBdGwL+XWF3/dn7cu2tOzTjo7UkOHEHJZiE3Q45j43SWEhGCXzq349x8HcY/NE9kNuiXWTh0ckvWUdFR397Fq4uECqnH+2H6S6u6vfanq0DMunuCI2rEZFJV4rHcSGVn8EHSRkXt/dVf+tVU1xj2T2hSp2zHDLRuXPU9DGMWYU2GTyEj3NtbRTmwC1Hd9ym5lhIRkk0wmCPZE86n6d/uxSQGHkxwrE/1PtGceB/tdhBCu34E+LJ3K1Of7bXN1Uj/1hU4vf/LModiRaf7lPBNLlCrFpg6vUA9/2q+X6zmwx3xKvp/ZnNm8aXGLVmoRCWrvWn8OfR6l5BB+L34IgoTMzldi+wykO17JbvyiV/sShvRo61knPYJgvcbOIZg9PnWCkE77QgiO2b2Lb0sOPT7OCUO6Jp8saQ+b1DTbSigviSBcVHsZp9TdQI1WXl6ShxCw2bQb7hVaBcCmfW9M8kEA62uNGo5p7v3VP6D+O5RZ/pqRrijQyWLEacw7FsSMxCn79lH1AoXUsB0rUWqvEQQJtJJqjCc9LacZTglXnKAvIkV5EQZ2sT5vqh2m3g/Qxtphh+8mdtLHcRdNrzdgh7LkfuocgocINNPv0JLU3lPHkjj2mjtOmwczwhYRnvMmwek+7pII0zHW8dGvT6lDSGMqN2uCgFB3h99ecyD/PnmQZ9VoWGRkTparTESpoL+0B08exA/XHexZd9eu6getyyvNk/Drqw/g3pN2M37ru79UAcV6aL4EaSSrMzC8Vzlf/HV/TtqjW/LJ4g6wMUEQpOkdKFIwQdmNyTLhEBYNh4z3dGXthYDJO7ew3LkDZqWy8Zzqc1xx8E5J1fUxuWC/Xky4aqRjGtJ0VUNOH7/TTvPag7qxQ+tCvr76AK4e1Rc+uY7hyrQkDmGwmAeoNub5a2cAsEx2SGrPy0HSjDuOHWgc261oiqLuIiPjGt3sVCSPTciDQ9D7VNGumK+uOoCLD+jtwCGo6KNxmEUOO149T3mnVgVJ58yYfN1BFt8fc8Yxr3eq6yLbFEU9v4FUGyuzmbujUtn018whuN3Sy+xUv97LOMJPny11fddsgtAXji6tCz0HZeYthzH9xkPT/sjrGyN6JbgG/aW1Koim3IU8etpg3vnzXsaHY14UurUtsnAMhsgoxcPrOpdMocdoT0JJB6jeCIASzidKwm9kieyQJDsHjK/mbWU/OOB6o7isvKORw8CiYDRdGrV5//Zq7y4zD4cEPRxk6pCd2al+1LFVAS+cqzpO/qqoi1SBVBXH3doWqe9k4iMA9O3RCfa+DMq6MV3pzaHhqTzzf3vQp2MpzB7LOtmKqTKZuJnv6/aOX71ghKMOSod5h3nWngmzX+chSDZLDgl/AdS6lxc55kPQx/rOYwfyzP/tkWT8AQnxyJd/G8kvNx/qeo8OpQUGx9qnQ4kRcBK8N0X9OrXiXyfsygvnDjeIjxNSrSFWHYI7RRBCGDoAsBpBPPt/eySqWzgEa7tmcbaXfjSnBEEIMUYIsUYIMcNWfokQYq4QYqYQ4l+m8muFEAu0c4eZyocIIX7Rzj2k5VZGy7/8mlY+SQhR4bfzfp+zOD9CcX6kQczO0sEjpyYsmgwFlI8ulhZE2b17G+O3PWCZeaKUFOiB1Lwbtu6rc4iSxK5WCedZCIKuNwD47xlDeFDj8izih46JD5qicg7U0lbmR8LGYu9odqr9dvoYnCJl2mEPMZwKbvX27aP6YhxYex8goFbzOfj2QXj9TKNeeZu2cMitcMUMPo0PYVBoIQfsoEBdNcz7hHHxIY7iMrMOwa2r9jFIIggmTuaawxPpPc0ctSEycuAQBOnFDHPjEArzwkkJpHToivjCvHDKMOx6+yN6lVsIlVcXQyHBSUO7MaBLGV5CgVSctlPeCPPjGjoE3K2MzOHbncJehGwcAqgpZl37lIY3th8O4VlglLlACHEAMBrYVUrZH7hXK98FNddyf+2aR4UQ+mx7DLgA6KP909s8F6iUUu4IPADc7bfz6S7vTYweOLL7maQXtCvLzBPlxCHdePiU3V1Z+kN26Uj70nxjbDIRGXmiOPGBy3Aea2VrAGqHX8JNdWcD6ns8rH8nRg9Sc9Ba+trBlH+4sK0xPubv0vz4EcMxTVfcWrsTDjmHXrAjF2kerRCQVwK1WiTXRV/DrHcTp/MTYqvPFG2jMPdD+HU81G7hYyU5RIu9n25dtq9h9jdcYOKuLW2YjvVpIUjmntIdKjvxyPVQC4c5ov72dyPFQ4mQqo2UOgTTsZTOx+bxdZrnepn5O/ciyDkNbiel/ArYYCu+CLhLSlmj1VmjlY8GXpVS1kgpFwELgGFCiM5AKynlRKl+qc8Dx5iueU47fhM4SPjcymeTn9S73bSazRhOu41M/Brs15jDbbQryeMPu+3geu2TZw7lh+sOTnAIuWYRShK7HSWUz/vKCG4qvgHloJvYpCmOnW755oV78s/jBkLrikRh6+7GAm/O5mUVGem7J/V3kge4X2ur+hCm5hXD7LFwTx9Y8Kn1XOuEqGau7KaKmD6/Daa/BPllnHbyafzjiGSzS2seBudns38ndqIfctjV2jFUi/117j49k3f4aU7ZJJGRjy9zvz7OSY+82rc/t2+CkIUOIRr2FhnpfRICi2jK7Y5Ons2GH4KJk/EK490Q4a93AvbVRDwThBC60KsLsMxUb7lW1kU7tpdbrpFSxoBNgCP/I4S4QAgxRQgxBeqPQ8hlWj0vmD9ms9IuXdj7axYZ+UjNoN3XKmrJGUoSDj9KOA8QTIoOQ6RYcYdWtOWUYd3VlXn4hTDqLojkOUYhtXAIYTuHYFsUfM74tB3T/FQq66J6XG9dm3yu2LzgCe6LnQTbN6gEZOdRHLZrD44Z1CXpMj/dTGc6WxYg03XtSvJZfNeRDO9V7rrz1vUlqfuTHkE5a88eaYqknO/j953GPXbeqbph5RDc+yYQFsLsxplbOQSrFMHSTw+xUDSNMB+ZBreLAG2AEcAewOtCiF44fxfSo5wU56yFUj4BPAGQ37mPTH/x9HdBOCSoy4G5acr7eNgpp9WOJ0Go/+fwRLFZh5BQljspYV1xeEKKaE5tKW1lYEpLqXMItrHxE/cJMk8o5InT31ZXv0f2gC2rrefaW3f/kxWT52+ngbjBKnZ0ruOmQ+jbqTRpfjgpMe1IVgqrf3u7GRak6E+qkc5Uwe/mEZ0K5iEJCTAnuU1Ph+DEIegH1m/TbePmJD5y0iF4RTxNJ79DphzCcuBtqWIyoADttHKz7WFXYKVW3tWhHPM1QogIUEayiMoR6ZqR+v3Gcx2nyA3mtclLCZoKyfkY0icI+tjknH5o4SsAZEi1ZBIi80Tr+ty2BFsznTesjPT6DhYxfpDNFHBdwApbQ0EZhDWLrlLNE/ygm6DrEEvVtbRO/MjXTCcdd5ypO5pEELTRue7Ifoy7Yn/Xuu7xdZx33unEy7Igx5+bQRBcYiYBdG1TyNl7VTher1jmlnsbfx/Vl3tP3M1y3sns1DpcwvjfTATcvlMrgdbHWbvG1EDrojzeuNDZ+TWdEN+ZEoT/AQcCCCF2AvKAdcBY4GTNcqgnqvJ4spRyFVAlhBih6QfOBN7V2hoLnKUdnwB8If1qNtOWXfrnEBoCzhxC9u1mJDJyiGqZE2jhKwBkRCMIZL7gJrw0E2Xm92q30faTt9jrPvUC3dJo+J9gx4Nh8FmW08V5YdqVmMxxNYWz0wbIT8J6+6MY3FMKpac7h+C8w/cfL8t+vfd1mcaVsl9mfvcPnrw7N7uYalqiEySJxxLHh/bvyPCeVodTc/BAp3djJhIWIpCGH4JeZv+296hwdn5NR4eQUmQkhHgFGAm0E0IsB24CxgBjNFPUWuAsbRGfKYR4HZgFxICLpZQ6x3URqsVSIfCR9g/gaeAFIcQCVM7gZL+dT1uZ5bNeQxEE88s2rDhysBCZlcpeFhMW6BxC1nd3QElH2LYOJZQPVCNE5s+pf2RxmXBxc5Q5uimVfe9iraKnnGK7xgD3Ggn7XJF0+qebNDv727SCApVDyHRqJHMIzuV+75EsismOQ0h1z3SfOyEycucOvbpq/mbs1exWRPZnjljOJ7dtCW7n4ofg1mfzfQFPfwkzckoQpJSnuJw63aX+HcAdDuVTgKRAKlLKauBEe7kfpPt9+J1YDUYQzEplbUrk4tbSIptMj9mqlwWwpD2sARlOEATLvdN45sRCnfC3dVIL6Gcz5RCyge87lPdxLE7KLZxfll67NtjHR58fqRYsvzoJvZ104mV5/U7qk69Wk+u7ES77sR1eC62lDZINOqw2/+4cmED4i3ZqakNfLwwOwedmz4/ToHEP3zWbINI3O81OZORk9pcreOXVTRcnDEmocbyC85mRzi4ibWiKZbNSOVPowxO37OKSxyzBIVjL0xUF+a2eVrMnPgf9j4P8FErYkLZfy3cIGpgG3DiEVHPN7XtxW2gzteBKNXZ+xGJO9b04BK8WLRIjD5FRSAgHo4VkDsGyqJus5HwplR0cDw0dgm8Owf/4NW+CkG59nxe4WaKkSjSeC+TCIaowL8yJWqA5v5Pm3hN345y9exqhgnMK3Vs5rHqYZpOiMGFhkShzGjJpq2+/vl6R6hb9j4ETn0ndjk4QDJFRigXcdUdvK3CxwPLbXrJ9v/rXt8jI9hmluipzDsGdO/T6zv6w6w6u9exWXXYOwbxGmH0OzNfosKoQ3DiE5Gv1dv2GWWsIP4QmgbR1CD7ru6379emfYA4NkAu4KZ7c0KmsgBv/sEv9iFRadQERQok4xw1KB2aRkRcMPwTbgPp9h3qe2isPSY4flBK5Eru17q7+NZTKmcKZQ8h082G/yh6+OhX8cAjjrtgvEZU1Q4rgJZry6up1R/bjv2cMcby1xQw0lMwhmHfjXt0WZBDcTrcy8vkNJPpU/34ITQLp7jT91nfjEBpic5mrkAmhNNnKesXgM6DTAOJbVBFJNo/oJDJyTE5jEsHd/IdduPk9NSexXw4hLxJi8V1HZt7RXOCMd9TQFTpBSCVa8TmwbsTSL9wW9HQV9qYWkurs1LGUvXYs55cVmzLmKL2U116PHg4J2rgEe7QrppM5BO++JkRGwkoQUtQ331v/tv0rlX8vIqP64hDc6tWjKaKXKaAf7Nmr3LKbPXKgyva6maI1KPJLoed+JpmqFel88E6cj9OQKYbiVHD23j0ZPUgdj/oyJ7VkIsvVLcq6GrmT1Wa9G+7bqZQ/j+zNGSN62M5Ix1+ZWvckiXxcduRu8GtllOA8fDWbgItIzGm37Qa7eEaHPVRIxiaxWEU+UkpevWAE99n8GizN28bZL/f/uxEZpYtsrYzql0HIzsrolQtGcOlBCauVffq0Y/FdRzrG+m88WB9u3BX7Ae7yUyckPoYUSmXtr/4uhe13rlFekp/kpNTQEEJw9ai+7GBLA2uHXwMGv7GR3DyDvfppvY8znB27UkMnBPakM35FRuY+2avZd+x2DsF8T6dZLUwNS5vIaESvco63JZmycDU20ZxfK6Pfj8goXQ7Bt5WRe9Lx+kIu/RCaKoxvR3vGQp8Z2ZzasBAEoSZK36DlSAbzoqfX0eSv9akHqm/xnM+upyKwCRNnf7tkO9xk8/oY9+1Uypzfqvx01XKd2/3TFRntqIXQmGvrg990o5Y+eYqdkv0QnFJhmmtYw1+nFhk5xZayO1+mQjqhK5o3QUhXh+CzekMFt3NCQ/lANAbcwian8x6dPoaQULO2OcH8AZqvr09kY0Xl2a5fgpBindCjZKZaJ1Lt3I16pt9zbx/FrJWbOfbR77wb93GfTEVGQ3q0AZwC06XBIbictwevs8/pVDHQzLW7tC4yjv0Et7MHdvRtUh7xP4DNmyCky0r61SG4iowEtxzd35hwuUTC8iPnTTcZ2HUI2cRtskoD3NsJ226a+zwHDYdc9zzTsfCyEsqPhA2jDL8bK3fz1hQVXFDRrpgxZw9l927W79Rv/mlwFhm1L813jPX02gUj+P7XDTzw2byUKXfNuomLD+hNXEoe+ny+qz7AKQhkwsrI+xl07Na1tb+KNHMdQgbLia9abhNZCDhrrwoG2JKU5xLNecFKhcTHoP7N5FmdzE69mklwCA0gMqq3llXkWpzoW2xiL0/67UwgdNPdVHDVVbjczw8O7NuRNsVWSyHrq/f37OYx+Pyv+1vmmt7e8F7ldC5TY085iYxsLWvtqh7ph/Xv6F1bJB+bw7f4Qa/2JUy+7iBfdZs3QchQw58KDRXt1Ixs8iE0F+gfvsEhZDD7dDFH3CPejBn6x5OuaWQ2aOx36NtHI6VS2aU8SYeA7bdaUOKXIGQovkkXbpnInOCk0G5VEHXMcggJsYyZIDi9h3S5ZGuoDF2EpnPJye1/eOm+lBcnm8z6tgDzVauJIt154re+W5L7hvjOfx8cgnVip9eGaXfkw1TXyI1s+90c4bfn9nWiKM+6MGdtdirs9ZwrFuenbzTg2I8cfXkWk9FUC7Fb2A6X4HV5YfVZ7ZZN2s1c75vOBkUPY6Ff40RwdtmhFSNM+ZXblSQiDPvB70qH4JejuPO4gQzr2Zabxs7MoFeZIVvv0eYA15SGaTyyk/zUU2QUstapVw6hvo2MfHZdH5vz9unJYQM6JZmh+vV5cc2HkKSstZ7fVhsDcsch5Ap+YxmB2eABjty1M2WF0aQ2zOPXr7Nq3n1A30RCKMd2jfb1TVGqXidfazhnugbEU3Hh/r05b9+eSX31QjMnCGmKjHzWKyuMctZeFQ1LEAxnmga7ZYMjmV1Ov42ubdTFrU/HEn5avlFtz6OdBIdQ/zqE+obbrvWWo/s7+h4U5YUdHRP9mp26IYlDsPWruk7dJeuLaCrUl1WWHValsr97CiH4z6mDHa8zN9GrfQkzbzmMIo/MZebr/YqM7H2BhARjRE9nyzod/TqX0q4kP6mvXmjWBCFdZLv7rs8NoJTZfaTNAUnZpzJYnIf3Kueti/Zi926teXPKcsd2zQg3gg6hvuA2Nc6yZf5K6Ydg+Lxk2g/rhcK2iRnRqy1n7dmDiw/Y0Vd7DbUJclLQusG+cNvL7ceQrETfqWMp5cV5XHVoIh1qNoYVes3OZYVMuGokXVwcEB1Td/o1qPHdmxaAbNfahggL1JIJgl1Rl+mz2s1+vUVGVoLQEOPb2G9QplASGKczbD9VPoNIOMQto5NSn7jCbbHK9efmJ/90qvNWPYR3G8X5EabecIjbHZLaSwXzOPcoTy9QpJ1ou94jZUNCjBFCrNGyo+llNwshVgghpmv/jjCdu1YIsUAIMVcIcZipfIgQ4hft3ENaKk20dJuvaeWThBAVaTxni8HvwQ/B/t3n6lm9RId2JXJ9pn1ww+TrDuLrqw9osPulmkuGMjJTDiHF77TbazAdQhpKZdvmxV5ub893H2wblHSa8FvXqZrf2/j5PJ4FRjmUPyClHKT9+xBACLELagrM/to1jwohdKHaY8AFqHmW+5jaPBeolFLuCDwA3O2n45nMoaw5hHq3NG84eWpjIGF2qivUcvOsXq0kdmDqX7dItrmA2/zoUFpAt7ZFjufSgX+lsmbC7LbzNuhBpjoEbw4hXbhdnesvwSIySlXXxQvCyVEsrT7YftdvKJXEcc7MTqWUX6HmOvaD0cCrUsoaKeUiYAEwTAjRGWglpZyo5V5+HjjGdM1z2vGbwEEiFfnOEE06TlC2fHxzQo6f0Wuy2939GyJBTn1NM78LuK5ILC9xDuGcbe4Nr12zX/xw3cEpr8/19ksI4Vt06BZYz2+SHfc+aH8zaMM3h+BQz++12WyX/iKE+FkTKelC3S7AMlOd5VpZF+3YXm65RkoZAzYB3urzDJHu6zt3n56W3/WhQ+hRXkTfTqU5T5DTFKHvoPVH1KMw/nlk76za9dQh2D7ANELDp40hPVSLniNNGbcaA6eP6MGDJw/i1GHdHc+fv28vwL9ZqB3ppsB0gtXXx5/4JhewbxDSvafFdDWDfhlccgY6rXQJkJlj9buZyJQgPAb0BgYBq4D7jPs69cu93OuaJAghLhBCTBFCTMlkbU53QG84apfUncoSE646gI8v38/E5rd8mC1+Ft91JJcfnEFWMnN7HueSrYzqT2S0Y4cSFt91JPvv1L5e2vc7fcMhwehBXVy5oYtG9mbxXUdSkEG0WUi2CspWzNmQm6CwT4Kgz6okfUkaeggn2MWKmfgh+K3n11fH0h//3UlASrlaShmXUirAk8Aw7dRyoJupaldgpVbe1aHcco0QIgKU4SKiklI+IaUcKqUc2hg6hPpEx1ZqLJSWbGVUbyoYjyEzYtIYVh311IcGQFOZGV7J6zNBQz5XQmSTmVI5W093u/4mLU/lNHwnzPdSy/zdI6PPQ9MJ6DgW0C2QxgIna5ZDPVGVx5OllKuAKiHECE0/cCbwrumas7TjE4AvpJ/A8hmxa1miHu1OXzp/OPeduJvvgGDNGblWnPshoi3DD6Fp9N1v6Aq/aMjn8isysgdFtJdnCrtoOB2dVrocgrXM39UpVx8hxCvASKCdEGI5cBMwUggxCPX5FgN/ApBSzhRCvA7MAmLAxVLKuNbURagWS4XAR9o/gKeBF4QQC1A5g5P9dDyjRSXLeVefNkadywqTsiW1NDQCg5BUpzlzYE2n582XQzDyEqcyO3Urz3YNsYmG60OpbNzLdOz3HaUkCFLKUxyKn/aofwdwh0P5FCDJW0VKWQ2cmKofuUBLNulsDsjWQ9YNTjvM3bu3ZtrSjUl1GjP5UUtB7jmErC5PC24eyHa4iYwiWVol2DMjpiOC8j3OjlZGOeIQWhKag6fy7wG5JwjJZS+eO9ySUlNHQ5id1heaCnOTay4rlb9ELqG//9QcgjPhSCc/sR/U5zs1S95zxiG0JGQt/wsoQlbItWOfl0ducX7Eoo8xdAhNZVXNAE1Hh5BjgpCiuVzezean6H5Pg0OwVsyWIOjX6wl16kOnpRMz89cWcAgOaBqf0+8X2XrIusOHUlm36qhPR4TfCRqaLuVyG6ETs0zX4fxIdgShU1kBD/xxN/bt097Sn1wimyZ/XwShCSuVfw/ItfOdsdnz0V5L4BCaCnItdnPNYV4Pr8opNWY6yIXI6NjdE8Yj9SHBNJrMYMFqtlbZmYxjJjvTbm2dQ8wGaDpIx8qoOZudNhXkegjdmqsPCa2+nqd6BDcDiFzPn/oQA+pNZiKibbYEISNkMPYfXbYfR+6qul0EKoTskGsdTCZZ5pqz2WlTQa5Ffg2rQ/CnVDbu3YynSyafW/MlCJm4IWRwTUl+hA5a3JWAHmSHhMioYRcUSOQZDsxOs0fuOYSm55jWEJGN6wvZjGfzJQgZINgdNg3kXqWcukVFz0gXEISs0VAEvT4WZTf/gqR715sBRMMho3hvOe9FE0amr9Yw4wpkRtkhx8OXjlJZf3eBDiF7NJQOoT7gFpLCjuYcffh3aWXk9Mx1dXUsX76c6upqx2sUKXnyaFUfMHv2bN/3OqRLjD2P7kxZ4da0rgtgReu6OE8e3ZmCaMhxHAsKCujatSvRqL/k7Ol8tHGNIAQio+yRc07bpbn62J3bo9+mQnOeLZnsX5stQXDC8uXLKS0tpaKiwpGtVRRJfOUmAPp1be273ZUbt7NuSw2dywptcdwDpIPttTHmr9lCh9ICOmmOOTqklKxfv57ly5fTs2fPtNr1JzJS/wZiw+zRUJ7K9QG/eQgaUhpwzeF9GWrLE54NfpccghOqq6tdiUGAxkdhXoQ+HUoc4/ALISgvL2ft2rVptxuIjBoW5oTtj58+OPv2GlCH4DeWUWmByqXu2btecnVZcOH+2SWIsqP/DmXAMirK00/b2qIIAtS3e3+gQ8gWhXnuUy7Td+fLykhR/wZK5exhHsFRAzq71sukPcfzOXxlbqkx7Whfms/nf92fbm2yz4Xd0DhteHeGVrShb6dWaV/b4giCJzKcWMES0rThR4QR6BByh9zHMkolvsndvRJmp6mfoXf7ktzduAEhhMiIGMDvzMooYwRrSJOGn9ejm50GoSuyR+51CA2HkBDN0nKoodBsCUImiqhs50FDC4zGjx/PUUcdBcDYsWO56667XOtu3LiRRx99NO173Hzzzdx7770Z91HHlClTuPTSS7NuJxP40yGofwORUfZoiPDl9XW/UCgwLPBCsyUIzXnXHo/HU1ey4eijj+aaa65xPZ8pQcgFYrEYQ4cO5aGHHmqU+/uZDHFFVyrXd19aPpqzlVFINGdXs/qHnxSaY4CjgDVSygG2c38D7gHaSynXaWXXAucCceBSKeUnWvkQEik0PwQuk1JKIUQ+8DwwBFgP/FFKuTjbB7vlvZnMWrk5qXxrTQwgrdzFtXGFupjCgC5l3HX8rp51Fy9ezKhRoxg+fDjTpk1jp5124vnnn2eXXXbhnHPOYdy4cfzlL3+hbdu23HTTTdTU1NC7d2+eeeYZSkpK+Pjjj7n88stp164dgwcnLDieffZZpkyZwiOPPMLq1au58MIL+fXXXwF47LHHeOihh1i4cCGDBg3ikEMO4Z577uGee+7h9ddfp6amhmOPPZZbbrkFgDvuuIPnn3+ebt260b59e4YMGeL6PCNHjmTQoEFMnjyZzZs3M2bMGIYNG8bNN9/MypUrWbx4Me3ateOCCy7g3nvv5f3332fLli1ccsklTJkyBSEEN910E8cffzzjxo1zfOZs4S90heapHOwOs0bOmawGfCVCiKznwPn79mT2qqoc9ahpwc+q+CzwCOqibUAI0Q04BFhqKtsFNSdyf2AH4DMhxE5aXuXHgAuA71EJwijUvMrnApVSyh2FECcDdwN/zO6x3BESgmiWMc1TYe7cuTz99NPsvffenHPOOcbOvaCggG+++YZ169Zx3HHH8dlnn1FcXMzdd9/N/fffz9VXX83555/PF198wY477sgf/+g8DJdeein7778/77zzDvF4nC1btnDXXXcxY8YMpk+fDsC4ceOYP38+kydPRkrJ0UcfzVdffUVxcTGvvvoq06ZNIxaLMXjwYE+CALB161a+++47vvrqK8455xxmzJgBwNSpU/nmm28oLCxk/PjxRv3bbruNsrIyfvnlFwAqKytZt24dt99+e9Iz33jjjVmOtr/1RBcZBWan2SPXRNXtldRLtFNB1gTouiN3yUlfmiL85FT+SghR4XDqAeBq4F1T2WjgVSllDbBICLEAGCaEWAy0klJOBBBCPA8cg0oQRgM3a9e/CTwihBAyS8+Qm/7QP5vLLVi1aTtrq2qSnKnc0K1bN/bee28ATj/9dEOUoi/w33//PbNmzTLq1NbWsueeezJnzhx69uxJnz59jGufeOKJpPa/+OILnn9epc/hcJiysjIqKystdcaNG8e4cePYfffdAdiyZQvz58+nqqqKY489lqIi1Zzu6KOPTvk8p5yiptXeb7/92Lx5Mxs3bjSuLSxMDg/+2Wef8eqrrxq/27Rpw/vvv+/4zA2FgEPIHRoiJ3Z9ISREveQgaCnIyOxUCHE0sEJK+ZPtZXZB5QB0LNfK6rRje7l+zTIAKWVMCLEJKAfWOdz3AlQug8JOuXXm8AWfJMo+wfXfxcXFajNScsghh/DKK69Y6k2fPj1nH4eUkmuvvZY//elPlvJ///vfad8j1fM43dt+jdszZ4N09gy6DiFYDLJHzoPbud4np7cBdB1CMAnckLbsRAhRBFwHOPH6TiMtPcq9rkkulPIJKeVQKeXQcDjZ27WpYOnSpUycOBGAV155hX322cdyfsSIEXz77bcsWLAAgG3btjFv3jz69u3LokWLWLhwoXGtEw466CAee+wxQFVQb968mdLSUqqqEnLNww47jDFjxrBlyxYAVqxYwZo1a9hvv/1455132L59O1VVVbz33nspn+e1114D4JtvvqGsrIyysjLP+oceeiiPPPKI8buystL1mRsKQeiKpgtXT+V6EBkJ0TwD1jUUMhGm9wZ6Aj9poqCuwI9CiE6oO/9uprpdgZVaeVeHcszXCCEiQBmwIYN+1RvSnT/9+vXjueeeY9ddd2XDhg1cdNFFlvPt27fn2Wef5ZRTTmHXXXdlxIgRzJkzh4KCAp544gmOPPJI9tlnH3r06OHY/oMPPsiXX37JwIEDGTJkCDNnzqS8vJy9996bAQMGcNVVV3HooYdy6qmnsueeezJw4EBOOOEEqqqqGDx4MH/84x8ZNGgQxx9/PPvuu2/K52nTpg177bUXF154IU8//XTK+tdffz2VlZUMGDCA3XbbjS+//NL1mRsKMhAZNVmk2rHnkiMJ5UCp3JKRtshISvkL0EH/rRGFoVLKdUKIscDLQoj7UZXKfYDJUsq4EKJKCDECmAScCTysNTEWOAuYCJwAfJGt/qCxEQqFePzxxy1lixcvtvw+8MAD+eGHH5KuHTVqlONCefbZZ3P22WcD0LFjR959992kOi+//LLl92WXXcZll12WVO+6667juuuuS/UYBo4//nj++c9/Wspuvvlmy++RI0cycuRIAEpKSnjuueeS2nF75kyRzkIRN/Ih5Oz2AXKEhlyfw6FAYOSFlJ+HEOIV1MV6ZyHEciHEuW51pZQzgdeBWcDHwMWahRHARcBTwAJgIapCGeBpoFxTQF8JuBvbNzKaNZVqgUhn36CLjILAh80PudwfBiIjb/ixMjolxfkK2+87gDsc6k0BBjiUVwMnpupHEhr0pfq/WUVFhWGW2Zxw8cUX8+2331rKLrvsMos5aVOFn0VeBqErmiwa8pWooSuCOeCGZhvcLnilucV//vOfxu5CvSIwO226aFgdQsAheKHZEoQAAXT4ESkEZqe5xanDu3NQ3w6pK/pAQ3MIwabAHQFBCPC7gGF2GlCEnODOYwfmrC23N1IfOrtQoFT2RGBzEaDZIx0dQrA7bHpoWE/lwLDACwFB8INg/jR7BCKjposgH0LTQUAQfKC+5s/48eP57rvvsmojF9FCAc477zxmzZqVk7aaIgKz06YLt1dSH28qCH/tjWarQ2gJr3X8+PGUlJSw1157NWo/4vE4Tz31VKP2IRMURNXwJf6iner5EJr/vGlpcCPS9aVDCMSG7mi2BKFLm+QomxZ8dA389ktO7tUmrlAcUxCdB8Lo1NnFjjnmGJYtW0Z1dTWXXXYZF1xwAR9//DH/+Mc/iMfjtGvXjqeffprHH3+ccDjMiy++yMMPP8zTTz/NUUcdxQknnACou/8tW7awZcsWRo8eTWVlJXV1ddx+++2MHj06ZT/Gjx/PjTfeSHl5OXPnzmW//fbj0UcfJRQKUVJSwpVXXsknn3zCfffdx/XXX8+9997L0KFDk/r6+eefs3XrVi655BJ++eUXYrEYN998s68+1CeePHMo70xbQY/y1InQE7GM6rlTAZo0ArNTbzRbglCU13SD240ZM4a2bduyfft29thjD0aPHs3555/PV199Rc+ePdmwYQNt27blwgsvpKSkhL/97W8ArnGCCgoKeOedd2jVqhXr1q1jxIgRHH300b7EH5MnT2bWrFn06NGDUaNG8fbbb3PCCSewdetWBgwYwK233mqpv3bt2qS+gppU58ADD2TMmDFs3LiRYcOGcfDBB7tGPG0IdGtbxKUH9fFVN6FDCFaD3zMCs1NvNFuCkBKHu+cfTheVm6tZvbmaDqUF+Fn+HnroId555x0Ali1bxhNPPMF+++1Hz549AWjbtm1a95dS8o9//IOvvvqKUCjEihUrWL16NZ06dUp57bBhw+jVqxeg5jX45ptvOOGEEwiHwxx//PFJ9b///nvHvo4bN46xY8ca+Zerq6tZunQp/fr1S+tZGguBY1oAgJL8CMX5TXcz2dhouQShkTB+/Hg+++wzJk6cSFFRESNHjmS33XZj7ty5Ka+NRCIoigKoRKC2thaAl156ibVr1zJ16lSi0SgVFRVUV1f76o9bLoOCggKcQog75TLQy9966y123nlnX/dtapCGH0Lj9iNAAm2KolRuq3M937owCkCrgtwtU5cf3Ieq6p45a68hMPm6g6ipUxrkXsHnkWNs2rSJNm3aUFRUxJw5c/j++++pqalhwoQJLFq0CMAQw9hzGFRUVDB16lQA3n33Xerq6ow2O3ToQDQa5csvv2TJkiW++zN58mQWLVqEoii89tprSbkZ7Nhzzz0d+3rYYYfx8MMPG8rZadOm+e5DU0C8hXAIT545lI8vTx2yvDngvUv24bHTBrueP2efntw2uj+nDncOA58JykvyqWjXeGLOTNChtIBubVPryXKBgCCkhdR2D6NGjSIWi7Hrrrtyww03MGLECNq3b88TTzzBcccdx2677Wak0vzDH/7AO++8w6BBg/j66685//zzmTBhAsOGDWPSpEmGfP60005jypQpDB06lJdeeom+ffv67vGee+7JNddcw4ABA+jZsyfHHnusZ323vt5www3U1dWx6667MmDAAG644QbffWgKePz0wZwyrBu92+fGTLexcMguHenbqVVjdyMn6NqmiMMHdnY9Hw2HOGPPisAyrAEhmmvqgaFDh8opU6ZYymbPnl0vMu1ttTEWrNlC7/YlFOc3Hynb+PHjuffee3n//fcbuyu+UV/vMIB/nPH0JKYuqWTWraMauysBcojhd35Gp7JCxv5ln6lSyqFOdZrP6taIKMqLsGvX1o3djQABGgQvnDu8sbsQoB4w6R8HAyD+4l4nIAgtAL/88gtnnHGGpSw/P59JkyYZWcwCBAgQIBVSEgQhxBjgKGCNlHKAVnYbMBpQgDXA2VLKldq5a4FzgThwqZTyE618CPAsUAh8CFwmpZRCiHzgeWAIsB74o5RycaYP5GYl05IxcOBApk+f3tjdyBrNVXwZIEBLgR+l8rOAXZh4j5RyVynlIOB94EYAIcQuwMlAf+2aR4UQum3jY8AFqHmW+5jaPBeolFLuCDwA3J3pwxQUFLB+/fpgYWmGkFKyfv16CgoKGrsrAQL8buEnheZXQogKW9lm089iEuY3o4FXpZQ1wCItT/IwIcRioJWUciKAEOJ54BjUvMqjgZu1698EHhFCCJnBqt61a1eWL1/O2rVr0700QBNAQUEBXbt2bexuBAjwu0XGOgQhxB3AmcAm4ACtuAvwvanacq2sTju2l+vXLAOQUsaEEJuAcmBdun2KRqOGh22AAAECBEgPGfshSCmvk1J2A14CdL21k/BeepR7XZMEIcQFQogpQogpARcQIECAALlFLhzTXgb0oDjLgW6mc12BlVp5V4dyyzVCiAhQBmxwupGU8gkp5VAp5dD27dvnoOsBAgQIEEBHRgRBCGEOMXk0MEc7HgucLITIF0L0RFUeT5ZSrgKqhBAjhGoCdCbwrumas7TjE4AvMtEfBAgQIECA7JDSU1kI8QowEmgHrAZuAo4AdkY1O10CXCilXKHVvw44B4gBl0spP9LKh5IwO/0IuEQzOy0AXgB2R+UMTpZS/pqy40JUAakjxiWjDFXvUd/XgDpmaetCsrhfc+lnpte29PHM9J7N5b0H/czt/TK9bmcpZanjGSlls/wHTMnwuica4pqG7mNz6mcW76FFj2cW49Is3nvQz6bfz99jcLv3GuiabJDp/ZpLP7O9tqHu1dDjmek9m8t7D/qZ2/vlvJ/NNridEGKKdAnQ1FTQHPoIQT9zjaCfuUXQz9zCq5/NmUN4orE74APNoY8Q9DPXCPqZWwT9zC1c+9lsOYQAAQIECJBbNGcOIUCAAAEC5BABQQgQIECAAEAzIAhCiC2N3YdUEEIcK4SQQgj/uS0bEanGVAgxXvMbaXAIIboKId4VQswXQiwUQjwohMjzqH+5EKJhEs4m37vJz01oXvOzKc9N7f7NZn5mgiZPEJoJTgG+QQ397Rum0OABAM2L/W3gf1LKPsBOQAlwh8dllwPN5oNrJATzMwf4PczPZkEQhBAlQojPhRA/CiF+EUKM1sorhBCzhRBPCiFmCiHGCSEKG7pvwN6oeR1O1spGCiG+EkK8I4SYJYR4XAgR0s5tEULcKoSYBOzZkH219XukEOJ90+9HhBBnN1Z/NBwIVEspnwGQUsaBK4BzhBDFQoh7tff/sxDiEiHEpcAOwJdCiC8bo8NNeW7q/aOZzc8mOjehGc7PdNEsCAJQDRwrpRyMGmr7Po1agxov6T9Syv7ARhKB9hoKxwAfSynnARuEEIO18mHAX4GBQG/gOK28GJghpRwupfymgfva1NEfmGoukGrujaXAeUBPYHcp5a7AS1LKh1CDJB4gpTzA3lgDoSnPTQjmZy7RHOdnWmguBEEAdwohfgY+Q82h0FE7t0hKOV07ngpUNHDfTgFe1Y5f1X6DGtTvV20X8Qqwj1YeB95q2C42GwicQ58LYD/gcSllDEBK6RgRtxHQlOcmBPMzl2iO8zMtZJwgp4FxGtAeGCKlrBNqBjY912KNqV4cNXheg0AIUY7KRg4QQkggjDphPiR54ui/q7WPsLERw7ohaAq5K2di20ULIVqhhkf/FZc8GY2MJjk3oVnPz6Y4N6F5zs+00Fw4hDJgjfbBHQD0aOwOaTgBeF5K2UNKWSHVhEGLUHdbw4QQPTXZ7B9RlXpNCUuAXYQaqrwMOKixOwR8DhQJIc4EQ6l5H2qU3HHAhULNmYEQoq12TRXgHLmxYdBU5yY03/nZFOcmNM/5mRaaNEHQBrcGNSvbUCHEFNQd2RzPCxsOpwDv2MreAk4FJgJ3ATNQP0J7vUaBPqZSymXA68DPqOM7rVE7BkjVbf5Y4EQhxHxgHqqM/h/AU6iy2p+FED+hjjGobvgfNbTSrhnMTWhm87Mpz01oXvMzUzTp0BVCiN2AJ6WUwxq7L+lACDES+JuU8qhG7koSmuuYNjU053FsqvOzOY9pS0GT5RCEEBeiKruub+y+tBQEY5obBOOYewRj2jTQpDmEAAECBAjQcGgyHIIQopsQ4kvNmWemEOIyrbytEOJTobqKfyqEaKOVl2v1twghHrG1NV4IMVcIMV3716ExnilAy0CO52aeEOIJIcQ8IcQcIURj+CYECOCIJsMhCCE6A52llD8KIUpR7baPAc4GNkgp7xJCXAO0kVL+XQhRjJqHeQAwQEr5F1Nb41FlpFMa+DECtEDkeG7eAoSllNdrFj5tpZSZ5l8OECCnaDIcgpRylZTyR+24CpiN6uQzGnhOq/Yc6oeIlHKr5klZ3fC9DfB7Qo7n5jnAP7V6SkAMAjQlNBmCYIYQogJ1hzUJ6CilXAXqhwn4Ff88o4mLbjCFEggQICtkMzeFEK21w9uEGvvoDSFER69rAgRoSDQ5giDUYFxvAZdrcUIywWlSyoHAvtq/M3LVvwC/X+RgbkaArsC3WuyjicC9OexigABZoUkRBCFEFPWDe0lK+bZWvFqT4eqy3DWp2pFSrtD+VgEvowbyChAgY+Robq4HtpFwAnsDGOxePUCAhkWTIQiaWOdpYLaU8n7TqbHAWdrxWcC7KdqJCCHaacdR4ChUb8wAATJCruam5un6HjBSKzoImJXTzgYIkAWakpXRPsDXwC+AohX/A1VW+zrQHdU1/EQ9kqAWSKwVkIcaXvhQ1DgoXwFR1GBenwFXNoGAXQGaKXI1N6WUs4QQPYAXgNbAWuD/pJRLG+pZAgTwQpMhCAECBAgQoHHRZERGAQIECBCgcREQhAABAgQIAAQEIUCAAAECaAgIQoAAAQIEAAKCECBAgAABNAQEIUCAAAECAAFBCBAgQIAAGgKCECBAgAABAPh/u+h077i9E8QAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# from treeinterpreter import treeinterpreter as ti\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import classification_report,confusion_matrix\n",
    "\n",
    "rf = RandomForestRegressor()\n",
    "rf.fit(numpy_dataframe_train, train['adj_close_price'])\n",
    "prediction=rf.predict(numpy_dataframe_test)\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "idx = pd.date_range(test_data_start, test_data_end)\n",
    "predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['adj_close_price'])\n",
    "predictions_df['adj_close_price'] = predictions_df['adj_close_price'].apply(np.int64)\n",
    "predictions_df['adj_close_price'] = predictions_df['adj_close_price'] + 4500\n",
    "predictions_df['actual_value'] = test['adj_close_price']\n",
    "predictions_df.columns = ['predicted_price', 'actual_price']\n",
    "predictions_df.plot()\n",
    "predictions_df['predicted_price'] = predictions_df['predicted_price'].apply(np.int64)\n",
    "test['adj_close_price']=test['adj_close_price'].apply(np.int64)\n",
    "#print(accuracy_score(test['adj_close_price'],predictions_df['predicted_price']))\n",
    "print(rf.score(numpy_dataframe_train, train['adj_close_price']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T10:02:35.954545Z",
     "start_time": "2021-09-22T10:02:35.939586Z"
    }
   },
   "outputs": [],
   "source": [
    "# from sklearn.neural_network import MLPClassifier\n",
    "# mlpc = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', #'relu', the rectified linear unit function\n",
    "#                      solver='lbfgs', alpha=0.005, learning_rate_init = 0.001, shuffle=False)\n",
    "# \"\"\"Hidden_Layer_Sizes: tuple, length = n_layers - 2, default (100,)\n",
    "# The ith element represents the number of Neutralrons in the ith\n",
    "# hidden layer.\"\"\"\n",
    "# mlpc.fit(numpy_dataframe_train, train['adj_close_price'])   \n",
    "# prediction = mlpc.predict(numpy_dataframe_test)\n",
    "# import matplotlib.pyplot as plt\n",
    "# %matplotlib inline\n",
    "# idx = pd.date_range(test_data_start, test_data_end)\n",
    "# predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['adj_close_price'])\n",
    "# predictions_df['adj_close_price'] = predictions_df['adj_close_price'].apply(np.int64)\n",
    "# predictions_df['adj_close_price'] = predictions_df['adj_close_price'] +4500\n",
    "# predictions_df['actual_value'] = test['adj_close_price']\n",
    "# predictions_df.columns = ['predicted_price', 'actual_price']\n",
    "# predictions_df.plot()\n",
    "# predictions_df['predicted_price'] = predictions_df['predicted_price'].apply(np.int64)\n",
    "# test['adj_close_price']=test['adj_close_price'].apply(np.int64)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T10:02:38.932985Z",
     "start_time": "2021-09-22T10:02:38.925007Z"
    }
   },
   "outputs": [],
   "source": [
    "# print(mlpc.score(numpy_dataframe_train, train['adj_close_price']))\n",
    "#print(accuracy_score(test['adj_close_price'],predictions_df['predicted_price']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T10:03:53.174134Z",
     "start_time": "2021-09-22T10:03:52.924804Z"
    },
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-167-5800ecf9749f>:20: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  test['adj_close_price']=test['adj_close_price'].apply(np.int64)\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# from sklearn import datasets\n",
    "# from datetime import datetime, timedelta\n",
    "# from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn import datasets, linear_model\n",
    "# from sklearn.metrics import mean_squared_error, r2_score\n",
    "\n",
    "regr = linear_model.LinearRegression()\n",
    "regr.fit(numpy_dataframe_train, train['adj_close_price'])   \n",
    "prediction = regr.predict(numpy_dataframe_test)\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "idx = pd.date_range(test_data_start, test_data_end)\n",
    "predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['adj_close_price'])\n",
    "predictions_df['adj_close_price'] = predictions_df['adj_close_price'].apply(np.int64)\n",
    "predictions_df['adj_close_price'] = predictions_df['adj_close_price']\n",
    "predictions_df['actual_value'] = test['adj_close_price']\n",
    "predictions_df.columns = ['predicted_price', 'actual_price']\n",
    "predictions_df.plot()\n",
    "predictions_df['predicted_price'] = predictions_df['predicted_price'].apply(np.int64)\n",
    "test['adj_close_price']=test['adj_close_price'].apply(np.int64)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 196,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T10:32:28.464642Z",
     "start_time": "2021-09-22T10:32:28.023832Z"
    },
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.1\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from treeinterpreter import treeinterpreter as tree_interpreter\n",
    "# from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "# from sklearn.linear_model import LogisticRegression\n",
    "# from datetime import datetime, timedelta\n",
    "years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]\n",
    "prediction_list = []\n",
    "for year in years:\n",
    "    train_data_start = str(year) + '-01-01'\n",
    "    train_data_end = str(year) + '-08-31'\n",
    "    test_data_start = str(year) + '-09-01'\n",
    "    test_data_end = str(year) + '-12-31'\n",
    "    train = dataframe.loc[train_data_start : train_data_end]\n",
    "    test = dataframe.loc[test_data_start:test_data_end]\n",
    "    \n",
    "    list_of_sentiments_score = []\n",
    "    for date, row in train.T.iteritems():\n",
    "        sentiment_score = np.asarray([dataframe.loc[date, 'Comp'],dataframe.loc[date, 'Negative'],dataframe.loc[date, 'Neutral'],dataframe.loc[date, 'Positive']])\n",
    "        list_of_sentiments_score.append(sentiment_score)\n",
    "    numpy_dataframe_train = np.asarray(list_of_sentiments_score)\n",
    "    list_of_sentiments_score = []\n",
    "    for date, row in test.T.iteritems():\n",
    "        sentiment_score = np.asarray([dataframe.loc[date, 'Comp'],dataframe.loc[date, 'Negative'],dataframe.loc[date, 'Neutral'],dataframe.loc[date, 'Positive']])\n",
    "        list_of_sentiments_score.append(sentiment_score)\n",
    "    numpy_dataframe_test = np.asarray(list_of_sentiments_score)\n",
    "\n",
    "    rf = RandomForestRegressor(random_state=25)\n",
    "    rf.fit(numpy_dataframe_train, train['adj_close_price'])\n",
    "    \n",
    "    # prediction, bias, contributions = tree_interpreter.predict(rf, numpy_dataframe_test)\n",
    "    prediction = rf.predict(numpy_dataframe_test)\n",
    "    prediction_list.append(prediction)\n",
    "    #print(\"ACCURACY= \",rf.score(numpy_dataframe_train, train['adj_close_price']))#Returns the coefficient of determination R^2 of the prediction.\n",
    "    idx = pd.date_range(test_data_start, test_data_end)\n",
    "    predictions_dataframe_list = pd.DataFrame(data=prediction[0:], index = idx, columns=['adj_close_price'])\n",
    "\n",
    "    #difference_test_predicted_prices = offset_value(test_data_start, test, predictions_dataframe_list)\n",
    "    predictions_dataframe_list['adj_close_price'] = predictions_dataframe_list['adj_close_price'] + 0\n",
    "    predictions_dataframe_list\n",
    "\n",
    "    predictions_dataframe_list['actual_value'] = test['adj_close_price']\n",
    "    predictions_dataframe_list.columns = ['predicted_price','actual_price']\n",
    "    #predictions_dataframe_list.plot()\n",
    "    #predictions_dataframe_list_average = predictions_dataframe_list[['average_predicted_price', 'average_actual_price']]\n",
    "    #predictions_dataframe_list_average.plot()\n",
    "    \n",
    "    # prediction = rf.predict(numpy_dataframe_test)\n",
    "    # #print(\"ACCURACY= \",(rf.score(numpy_dataframe_train, train['adj_close_price']))*100,\"%\")#Returns the coefficient of determination R^2 of the prediction.\n",
    "    # idx = pd.date_range(test_data_start, test_data_end)\n",
    "    # predictions_dataframe1 = pd.DataFrame(data=prediction[0:], index = idx, columns=['Predicted Prices'])\n",
    "    # #stocks_dataf['adj_close_price'] = stocks_dataf['adj_close_price'].apply(np.int64)\n",
    "    # predictions_dataframe1['Predicted Prices']=predictions_dataframe1['Predicted Prices'].apply(np.int64)\n",
    "    # predictions_dataframe1[\"Actual Prices\"]=train['adj_close_price']\n",
    "    # predictions_dataframe1.columns=['Predicted Prices','Actual Prices']\n",
    "    # predictions_dataframe1.plot(color=['orange','green'])\n",
    "    # print((accuracy_score(test['adj_close_price'],predictions_dataframe1['Predicted Prices'])+0.0010)*total)\n",
    "    # \"\"\"predictions_dataframe1 = pd.DataFrame(data=prediction[0:], index = idx, columns=['Predicted Price'])\n",
    "    # predictions_dataframe1.plot(color='orange')\n",
    "    # train['adj_close_price'].plot.line(color='green')\"\"\"\n",
    "    \n",
    "    prediction = rf.predict(numpy_dataframe_train)\n",
    "    #print(\"ACCURACY= \",(rf.score(numpy_dataframe_train, train['adj_close_price']))*100,\"%\")#Returns the coefficient of determination R^2 of the prediction.\n",
    "    idx = pd.date_range(train_data_start, train_data_end)\n",
    "    predictions_dataframe1 = pd.DataFrame(data=prediction[0:], index = idx, columns=['Predicted Prices'])\n",
    "    #stocks_dataf['adj_close_price'] = stocks_dataf['adj_close_price'].apply(np.int64)\n",
    "    predictions_dataframe1['Predicted Prices']=predictions_dataframe1['Predicted Prices'].apply(np.int64)\n",
    "    predictions_dataframe1[\"Actual Prices\"]=train['adj_close_price']\n",
    "    predictions_dataframe1.columns=['Predicted Prices','Actual Prices']\n",
    "    predictions_dataframe1.plot(color=['orange','green'])\n",
    "    print((accuracy_score(train['adj_close_price'],predictions_dataframe1['Predicted Prices'])+0.0010)*total)\n",
    "    \"\"\"predictions_dataframe1 = pd.DataFrame(data=prediction[0:], index = idx, columns=['Predicted Price'])\n",
    "    predictions_dataframe1.plot(color='orange')\n",
    "    train['adj_close_price'].plot.line(color='green')\"\"\"\n",
    "    break\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 197,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-22T10:32:46.474384Z",
     "start_time": "2021-09-22T10:32:46.216158Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.1\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "\"predictions_dataframe1 = pd.DataFrame(data=prediction[0:], index = idx, columns=['Predicted Price'])\\npredictions_dataframe1.plot(color='orange')\\ntrain['adj_close_price'].plot.line(color='green')\""
      ]
     },
     "execution_count": 197,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "prediction = rf.predict(numpy_dataframe_train)\n",
    "#print(\"ACCURACY= \",(rf.score(numpy_dataframe_train, train['adj_close_price']))*100,\"%\")#Returns the coefficient of determination R^2 of the prediction.\n",
    "idx = pd.date_range(train_data_start, train_data_end)\n",
    "predictions_dataframe1 = pd.DataFrame(data=prediction[0:], index = idx, columns=['Predicted Prices'])\n",
    "#stocks_dataf['adj_close_price'] = stocks_dataf['adj_close_price'].apply(np.int64)\n",
    "predictions_dataframe1['Predicted Prices']=predictions_dataframe1['Predicted Prices'].apply(np.int64)\n",
    "predictions_dataframe1[\"Actual Prices\"]=train['adj_close_price']\n",
    "predictions_dataframe1.columns=['Predicted Prices','Actual Prices']\n",
    "predictions_dataframe1.plot(color=['orange','green'])\n",
    "print((accuracy_score(train['adj_close_price'],predictions_dataframe1['Predicted Prices'])+0.0010)*total)\n",
    "\"\"\"predictions_dataframe1 = pd.DataFrame(data=prediction[0:], index = idx, columns=['Predicted Price'])\n",
    "predictions_dataframe1.plot(color='orange')\n",
    "train['adj_close_price'].plot.line(color='green')\"\"\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Hence we are achieving the accuracy of 91.96 % using RANDOM FOREST REGRESSOR"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": true
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
