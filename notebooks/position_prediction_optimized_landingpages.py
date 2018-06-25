# gist -u aea9fb724ead3674a20eb27b2dfe0237 -p position_prediction_optimized_landingpages.py
# inserire ache diffbotapi module nella cartella helper_classes
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
from helper_classes.semrush.semrush_api import SEMRushAPI
from helper_classes.diffbot.diffbotapi import Diffbot


# your SEMRush API Key
YOUR_DIFFBOT_DEV_TOKEN = '5888150e501cca562ef76a5bbbffaaa0'
diffbot = Diffbot(YOUR_DIFFBOT_DEV_TOKEN)

# your SEMRush API Key
api_key = ''
# SEMRush Keyword Database used (Complete list https://www.semrush.com/api-analytics/)
database = 'it'

sr = SEMRushAPI(api_key, database=database)
reg_score = []


# In[2]:


# Define Domain to optimize & competitors (more competitors=better results)
main_domain = ['www.borsaitaliana.it']
competitor_domains = ["www.unicredit.it"]
base_domains = main_domain + competitor_domains


# In[4]:


# Get Information about the Domains (Quelle: SEMRUSH)
domain_arr = []
for domain in base_domains:
    sr_result = sr.get_domain_overview(domain)
    if sr_result:
        domain_arr.append(sr_result[0])


# In[5]:


domain_df = pd.DataFrame.from_records(domain_arr)


# In[ ]:


domain_df


# In[ ]:


# Get all Keywords for the listed Domains (Quelle: SEMRUSH API)
# This can take a while an need API Credits
# The results of the API will be cached
keyword_df = None
keyword_df = pd.DataFrame()
for domain in base_domains:
    res = sr.get_domain_keywords(domain, count=20)
    if res:
        kw_df = pd.DataFrame.from_records(res)
        kw_df["Domain"] = domain
        keyword_df = pd.concat([kw_df, keyword_df])


# In[8]:


# Delete not needed columns
if "Position Difference" in keyword_df:
    del keyword_df["Position Difference"]
if "Previous Position" in keyword_df:
    del keyword_df["Previous Position"]
if "Trends" in keyword_df:
    del keyword_df["Trends"]
if "Traffic (%)" in keyword_df:
    del keyword_df["Traffic (%)"]
if "Traffic Cost (%)" in keyword_df:
    del keyword_df["Traffic Cost (%)"]

# Convert Strings to numbers
keyword_df["CPC"] = pd.to_numeric(keyword_df["CPC"]).astype(float)
keyword_df["Competition"] = pd.to_numeric(keyword_df["Competition"]).astype(float)
keyword_df["Search Volume"] = pd.to_numeric(keyword_df["Search Volume"]).astype(float)
keyword_df["Position"] = pd.to_numeric(keyword_df["Position"]).astype(float)
keyword_df["Number of Results"] = pd.to_numeric(keyword_df["Number of Results"]).astype(float)


# In[9]:


keyword_df.sort_values('Search Volume', ascending=False).head(10)


# In[10]:


# Each keyword only once
keyword_df_unique = keyword_df.drop_duplicates(subset='Keyword', keep="last")

# Delete not neded rows
del keyword_df_unique["Position"]
del keyword_df_unique["Url"]
del keyword_df_unique["Domain"]

print("Count unique Keywords for all Domains: {}".format(len(keyword_df_unique.index)))
keyword_df_unique.head(10)


# -------------

# In[11]:


# Get Information about the found urls of the positions
# Reduce to only competitors
all_urls = keyword_df.drop_duplicates(subset='Url', keep="last")

urls_to_crawl_df = all_urls[all_urls['Domain'].isin(base_domains)]
urls_to_crawl = urls_to_crawl_df.Url
print("Count of URLs to crawl: {}".format(len(urls_to_crawl)))


# In[12]: CRAWLER DIFFBOT


# =============================================================================
# # load a simple crawler to fetch URLs
# from helper_classes.simple_crawler.simple_crawler import Crawler
# crawler = Crawler(30) # How much Threads in parallel?
#
# # Crawl
# # Crawl results are cached
# res = crawler.get_websites(urls_to_crawl)
#
# # Build DF from cralwed URLS
# crawled_df = pd.DataFrame.from_records(res)
# crawled_df.rename(columns={'url': 'Url'}, inplace=True)
#
# =============================================================================

d=[]
for item in urls_to_crawl:
    response=diffbot.get_article({'url': item,})
    print("extracting:",item)
    d.append({'response_content': response["objects"][0]["text"].lower(),
            'response_status': 200,
            'title':response["objects"][0]["text"].lower(),
            'Url':item})

crawled_df=pd.DataFrame(d)


# In[13]:


print("Title & Content for {} Urls".format(len(crawled_df.index)))
crawled_df.head()


# In[14]:


# Combine Keywords and positions with crawled content
combined_df = pd.merge(keyword_df, crawled_df, on='Url', how='outer')


# In[15]:


combined_df.head()


# In[16]:


# Check each Title, URL, Content if words of Keyword phrase are in it and save information as float
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from urllib.parse import urlparse
stopwords_list = set(stopwords.words('italian'))

def check_keyword(args):

    words = args["Keyword"].split(" ")
    words_wo_stop = [i for i in words if not i in stopwords_list]
    count_kw = len(words_wo_stop)

    found_words_title = 0
    found_words_content = 0
    found_words_url = 0
    found_words_domain = 0
    url_found = 0.0
    content_found = 0.0
    title_found = 0.0
    domain_found = 0.0
    is_homepage = 0
    for word in words_wo_stop:
        if str(word) in str(args["Domain"]):
            found_words_domain += 1

        path = urlparse(args["Url"]).path
        if str(word) in str(path):
            found_words_url += 1
        if "/" == path:
            is_homepage = 1
        if str(word) in str(args["title"]):
            found_words_title += 1
        if str(word) in str(args["response_content"]):
            found_words_content += 1
    if found_words_url > 0:
        url_found = found_words_url / count_kw
    if found_words_title > 0:
        title_found = found_words_title / count_kw
    if found_words_content > 0:
        content_found = found_words_content / count_kw
    if found_words_content > 0:
        domain_found = found_words_domain / count_kw
    return pd.Series({'title_found': title_found,
                      'content_found': content_found,
                      'url_found': url_found,
                      'domain_found': domain_found,
                      'is_homepage': is_homepage})

rated_df = combined_df.join(combined_df.apply(check_keyword, axis=1))


# In[17]:


rated_df.head()


# In[18]:


# Set an ID for each domain and keyword so we can provide the ML Algo the ID and have a representating array
domains = list(rated_df.Domain.unique())
keywords = list(rated_df.Keyword.unique())
def get_domain_id(domain):
    return domains.index(domain)
def get_keyword_id(keyword):
    return keywords.index(keyword)
rated_df['Domain_id'] = rated_df['Domain'].apply(get_domain_id)
rated_df['Keyword_id'] = rated_df['Keyword'].apply(get_keyword_id)


# In[19]:


# Show example entrys for Keyword ID 2
rated_df[rated_df["Keyword_id"]==2].head()


# In[20]:


# get only the top 40 results
# rated_df_top40 = rated_df[rated_df["Position"]<40]


# In[21]:


# Delete all rows with no response_status (if any)
rated_df.dropna(subset=['response_status'], how='all', inplace=True)
print("{} rows for machine learning".format(len(rated_df.index)))


# In[#  ¡Define Features (which rows should the ml algorithm should consider for learning)

features = ["Domain_id",
            "content_found",
            "domain_found",
            "title_found",
            "url_found",
            "Keyword_id",
            "CPC",
            "Number of Results",
            "Search Volume"]
# Define Target (What should be predicted)
target = "Position"


# In[#   Split in train and test data
train = rated_df.sample(frac=0.8)
test = rated_df.loc[~rated_df.index.isin(train.index)]
print ("Train rows: {}".format(len(train.index)))
print ("Test rows: {}".format(len(test.index)))


# In[#    Train Model with Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

sc = StandardScaler()

mlc = MLPClassifier(activation = 'relu', random_state=1,nesterovs_momentum=True)
loo = LeaveOneOut()
pipe = make_pipeline(sc, mlc)

# Train the Model and check wich of the Parameters works best
parameters = { "mlpclassifier__hidden_layer_sizes":[(300,),(500,)],
              "mlpclassifier__solver" : ( "sgd", "lbfgs"),
              "mlpclassifier__max_iter": [500, 1000, 2000],
              "mlpclassifier__learning_rate_init":[0.001, 0.1]  }
MLPClassifierModel = GridSearchCV(pipe, parameters,n_jobs= -1,cv = 5)
MLPClassifierModel.fit(train[features], train[target])

# Save Model to file to used it later
file = open("test3_k_t_o_MLPClassifierModel_10_comp.pkl", 'wb')
pickle.dump(MLPClassifierModel, file)
file.close()


# In[# Open Model from File (could be skipped if you've trained it just a second before ;) )
file = open("test3_k_t_o_MLPClassifierModel_10_comp.pkl", 'rb')
MLPClassifierModel = pickle.load(file)
file.close()


# In[# rated_df[rated_df["Domain"].isin(main_domain)]


# In[#   Get Data to query the Model

id = 35 # just change ID to one of the row ids from above
data = [
    rated_df.iloc[id]["Domain_id"],
    rated_df.iloc[id]["content_found"],
    rated_df.iloc[id]["domain_found"],
    rated_df.iloc[id]["title_found"],
    rated_df.iloc[id]["url_found"],
    rated_df.iloc[id]["Keyword_id"],
    rated_df.iloc[id]["CPC"],
    rated_df.iloc[id]["Number of Results"],
    rated_df.iloc[id]["Search Volume"]
    ]



# In[# test new data

# =============================================================================
# in questo esempio c'è il modello di attribuzione: si introduce il nuovo testo, e si estraggono le features e
# e si ottiene
#    - keyword ID
#    - domain ID
#    - altre features
#per esempio uin array così:
#
#data = [1,1,0,1,0,23,0.32,795000,15]
# =============================================================================




# In[28]:


# What Data we query?
def print_data(data):
    print("Domain: {}".format(domains[int(data[0])]))
    print("Keyword: {}".format(keywords[int(data[5])]))
    print("Keyword in Content Found? {}".format(data[1]))
    print("Keyword in Title Found? {}".format(data[3]))
    print("Keyword in Domain Found? {}".format(data[2]))
    print("Keyword in URL Found? {}".format(data[4]))
    print("CPC: {}".format(data[6]))
    print("Number of Results: {}".format(data[7]))
    print("Search Volume: {}".format(data[8]))
    #print("Position: {}".format(data[9]))
print_data(data)


# In[29]:


# Predict Position
df_to_predict = pd.DataFrame(data = [data], index=[0], columns=features)
res = MLPClassifierModel.predict(df_to_predict)
print("MLPClassifierModel predicted Position:  {}".format(int(res[0])))


# In[30]:


# Modify Data for that Keyword
modified_data = data
data[1] = 1
data[3] = 1
print_data(modified_data)


# In[31]:


# Predict Position
df_to_predict = pd.DataFrame(data = [modified_data], index=[0], columns=features)
res = MLPClassifierModel.predict(df_to_predict)
print("MLPClassifierModel predicted Position:  {}".format(int(res[0])))


# In[32]:


# Now we want to predict Positions for all Keywords if title & Description are optimized for one of the Domains


# In[33]:


# Get a list with all Keywords and if any with position and found Url
predict_df = pd.merge(keyword_df[keyword_df["Domain"].isin(main_domain)][["Position", "Keyword", "Url"]],keyword_df_unique, on='Keyword', how='outer')


# In[34]:


predict_df.head()


# In[35]:


domain_id = get_domain_id(main_domain[0])
# Predict Data for each row
def optimized_position(args):
    try:
        keyword_id = get_keyword_id(args["Keyword"])
    except:
        return pd.Series({'optimized_position': None })
    data = [
        domain_id,
        1,
        0,
        1,
        0,
        keyword_id,
        args["CPC"],
        args["Number of Results"],
        args["Search Volume"]
    ]
    df_to_predict = pd.DataFrame(data = [data], index=[0], columns=features)
    res = MLPClassifierModel.predict(df_to_predict)
    return pd.Series({'optimized_position': int(res[0])})

final_df = predict_df.join(predict_df.apply(optimized_position, axis=1))


# In[36]:


new_result = final_df.sort_values('Search Volume', ascending=False)[final_df["optimized_position"]<10]


# In[37]:


# Show Keywords where the Model predicts a top 10 Position
new_result


# In[38]:

print("------>Show Keywords where the Model predicts a top 10 Position")
new_result.to_csv("result.csv")
