## Query Evaluation

### Successful Queries
The implemented search system was tested on different levels of queries:
* Easy Queries:
    Single-word queries
    * Statues
    ![image info](images/statue.png)

* Medium Queries:
    Slightly descriptive queries
    * Food in a box
      ![image info](images/food_in_box.png)

* Hard Queries:
    More descriptive queriess
    * Women standing next to a car
    ![image info](images/women_car.png)
   
### Unsuccessful Queries


### Keyword vs Context Trade-off

During the evaluation of our text2image search system, two test cases revealed an interesting discrepancy in search results:
#### Case # 1

When searching for 'vintage fashion more than one ladies' the system yielded accurate results. However, when the query was simplified to 'fashion more than one ladies,' the system did not provide as accurate of the results.
![image info](images/vintage_fashion.png)
![image info](images/fashion_ads.png)

This observation suggests that the system's performance is more keyword-based rather than accurately capturing the underlying intent of the query. 

#### Case # 2:
In the second case, a random German word was used as a query. Surprisingly, the system was able to retrieve pictures that either contained German text or were related to Germany (Porsche).

![image info](images/german.png)

This observation contrasts with our earlier findings, where the system appeared to prioritize specific keywords over contextual understanding. 

Overall , It suggests that the system's performance may vary depending on the complexity and specificity of the query. As queries become more descriptive, the system may exhibit a tendency to focus more on specific keywords rather than comprehensively interpreting the user's intent.

## Method of Quantitative Evaluation of Retrieval Accuracy:

Following things can be considered during quantitative evaluations:
* Data Labelling:

    Ensure accurate labelling of each image with details like brand names and product categories. Consider incorporating sentiment analysis for better understanding of user queries.

* Query Preparation:

    Instead of relying solely on predefined queries, consider incorporating user feedback to generate queries dynamically. This approach can mimic real-world search scenarios more accurately and provide insights into user intent that may not be captured by static queries.
   

* Evaluation Metrics:

   Besides traditional metrics like Precision and Recall, incorporate user feedback on result relevance to continuously improve relevance judgment based on user interactions.