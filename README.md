# Airline-Company-Clustering-and-ANN-model-predicting-Cusomters-Satisfaction

## Problem
- An airline company would like to know more about their customers and how to please them. They have provided some information about 20,000 customers as well as the results of a satisfaction survey. The data is available here.

They would like you to:
1. Segment the customers into groups and describe those groups, what they have in common and how they differ.

2. Model the data to see if overall customer satisfaction can be predicted by information about their flights and answers to the survey other than overall satisfaction.

## Clustering Instructions:
The airline company would like you to segment the customers and create an analytical report on the clusters describing each group and describing how they are different.

1. Load the dataset.

2. Explore the data.

3. Prepare the data for modeling.

4. Use a KMeans model to cluster the passengers into an optimal number of clusters.

  - Use an elbow plot and silhouette score to find the optimal number of clusters.
  - There must be between 2 and 10 clusters.
5. Analyze and report on the clusters.

  - Describe each cluster.
  - How are the clusters different?
  - Create 2 report quality explanatory visualizations showing important differences between the clusters.
  - Interpret and explain each visualization.
## Modeling Instructions:
The airline would also like to know how well overall satisfaction can be predicted from information about passengers and satisfaction with specific parts of their experience. Your target will be the 'satisfied' column

- Use random_state = 42 for your train_test_split
- Use PCA to prepare the data for modeling.
- Reduce the number of features in the data
- Be sure to use the PCA-transformed data when you fit and validate your predictive model.
- Do not leak data while preparing your data for modeling.
- Use a deep learning model to predict whether customers will report that they were satisfied.
- Use the PCA-transformed data you created in Step 2.
- Create 3 different versions of a sequential model.
- Each new model should be an iteration of the previous model.
- Justify in writing why you changed what you did for each new model. (For example, why did you add layers, regularization, nodes, etc)
- Adding or reducing epochs does not count as a new model.
- Use some form of regularization with at least one model.
- Evaluate each model with multiple appropriate metrics.
- Choose a final model and justify your choice.
- Evaluate your final model with multiple metrics.
- Based on those metrics, explain in writing how well your model will solve the business problem.

## Clustering

![silhoutte and inertia](https://github.com/user-attachments/assets/4685fde5-f10b-47af-8611-63ff744df66d)

Regarding the silhoutte score 2 and 4 are good number of clusters, however regarding the inertia there's an elbow at 3,5,7,8. To combine both inertia and silhoutte score im goin to choose 4 clusters


![clusters](https://github.com/user-attachments/assets/c0c36ca7-5102-4266-9fa3-d3a1b98bc021)

Cluster 2 has the highest age, cluster 1 has the highest class, longest flights, best wifi services, most ease of online booking, best gate location, online boarding, most comfort seats, most inflight entertainment, best on board service, biggest leg room, best baggage handling, best checkin service, inflight service, most business travels and most satisfied customers

![clusters boxplot](https://github.com/user-attachments/assets/4e04f924-e51e-4a89-803b-04ed93151e06)

from this visualization we proove our point that cluster 1 are high class customers and cluster 3 is for economic class customers

## ANN Modeling

### Baseline Model
![accuracy best model](https://github.com/user-attachments/assets/61fa3920-f653-41ae-87ac-e6cedf0d893c)

![loss best model](https://github.com/user-attachments/assets/9b215a6b-4582-4eee-9096-334faaf20a49)

![precision best model](https://github.com/user-attachments/assets/a4c3febf-cbba-4531-a675-627e49d33ec3)

![recall best model](https://github.com/user-attachments/assets/ace76008-9a0a-47e1-8b95-49b18cab9a0c)

the baseline needs 20 epochs to learn, the model is not bad it can be better, it has a small overfitting, good loss ratio but can be a bit improved, as a first step we will add a dropout of .25 and reduce the epochs to 20

![train metrics](https://github.com/user-attachments/assets/6a6f1adc-cacd-4633-966a-d286674a8b8b)

![cm train](https://github.com/user-attachments/assets/34c7d7bf-6f18-4022-9fb6-84b0c982bc0e)


![test metrics](https://github.com/user-attachments/assets/7fbc7887-2b7d-40e1-acb3-0e2d82093c9a)

![cm test](https://github.com/user-attachments/assets/0c96a7b2-d5ee-4223-8457-c1e7b8ead240)

Regarding the confusion matrix the baseline is doing pretty well and no overfitting apears but lets assume there's overfitting and add a dropout

### ANN model with 0.25 dropout

![model with dropout acc](https://github.com/user-attachments/assets/977eb84d-fd10-4a08-b658-a7254c3fce3a)


![model with dropout rec](https://github.com/user-attachments/assets/53c1dbdc-e6c7-42c3-91b6-1033585679f2)

![model with dropout pre](https://github.com/user-attachments/assets/69244afe-e7b6-4f62-a334-a424162ec982)

Regarding the loss it seems it has less overfitting than the baseline model, but to improve it better lets run a hyperparameters tuning to get the best model

![train metrics 2](https://github.com/user-attachments/assets/c00d736b-e36b-4c3c-9bc0-23c5c561dedf)

![cm train 2](https://github.com/user-attachments/assets/54194a20-690e-44d8-b9f7-7f634553b415)

![test metrics 2](https://github.com/user-attachments/assets/8d4af72f-046d-488d-afc7-439adee18ddb)



so we had a better accuracy with the basline model, however we have less false positives in this model but more false negatives
### ANN with Hyperparameter tuning
![model hp](https://github.com/user-attachments/assets/bf92fe47-a7ce-4e0e-95b6-5cccb5ec1a75)

![model hp loss](https://github.com/user-attachments/assets/4a6b77ba-d371-43ea-b08f-8efa8522d4cb)



![metric train 3](https://github.com/user-attachments/assets/8798b1d3-40a5-4fc9-86ac-49d7f404bc2c)

![cm train 3](https://github.com/user-attachments/assets/7819ce3a-8042-493b-bd0f-4e9e33abe765)

![metric test 3](https://github.com/user-attachments/assets/f98ec625-483b-463a-b844-5191d62f5819)

![cm test 3](https://github.com/user-attachments/assets/4b2a26c1-efc2-4825-a7cc-f6f7fbb98583)

after having the 3 different models the baseline model is the best model because it has more true predictions than the other 2 models, also higher accuracy no overfitting when checking out the confusion matrix but when checking the loss graph you will find a difference between the training data and the validation

The model will have 90% true predictions which from 100 customers, 10 of them the model wont be able to predict correctly regarding their satisfaction, also the errors that the model has mostly  by predicting that the customers are not satisfied by the service which means it will help the company to satisfy more customers or even the satisfied customers make them more satisfied
