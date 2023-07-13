# <a name="top"></a>TELCO CHURN 
![]()

by: Alfred W. Pirovits

<p>
  <a href="https://github.com/Alfred-W-S-Pirovits-Jr/telco_churn_project#top" target="_blank">
    <img alt="" src="" />
  </a>
</p>


***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___

<img src="https://docs.google.com/drawings/d/e/2PACX-1vR19fsVfxHvzjrp0kSMlzHlmyU0oeTTAcnTUT9dNe4wAEXv_2WJNViUa9qzjkvcpvkFeUCyatccINde/pub?w=1389&amp;h=410">

## <a name="project_goals"></a> Project Goals:
- Explore the enormous dataset to find patterns
- Organize features creating dummies for the fips code to represent Los Angeles, Orange, and Ventura Counties
- Run analysis to see what features are correlated/dependent to the target tax assessed value of the homes
- Try and determine which factors are driving price 
- See if we can find the driving factors that can build a Robust Model

## <a name="project_description"></a>Project Description:
[[Back to top](#top)]
The purpose of this project is to determaine the main drivers of home value for Single Family Residences in the Los Angeles, Orange and Ventura Counties and build Regression Model based on features correlated with the assessed tax value of the home.  Since we are looking at home sales in 2017 the tax assessed home values are sales price of the home in question.  With each year that passes since the sale of a home in California the assessed value and actual home value diverge due to Proposition 13 passed in 1978 which caps the tax value increase at 2% in any given year.
***
## <a name="planning"></a>Project Planning:    
[[Back to top](#top)]
The main goal of the project was to explore the data presented and see what I could discover.  Since there was a lot of data to go through, the main plan of the project was one of discovery.  After discovering relationships I started to hone in on how to analyze the dataset.  I found the features most correlated with the tax value and used those columns to build Regression Models.



        
### Hypothesis
There were many hypotheses, however the general meta hypothesis was whether or not the tax assessed value of the home was dependent on each of  the features chosen one by one.  The general Null hypothesis was that the given feature and the target (home tax assessed value) were independant while the Alternate Hypothesis was that they were dependent.  An alpha of 0.01 was chosen given the number of observations there were and the fact that there were so many features to choose from to analyze.  



### Target variable
The target variable is tax_amount.  


### Need to haves (Deliverables):
Github repo with the following:

1. Readme (.md)
- Project goals
- Project description
- Project planning (lay out your process through the data science pipeline)
- Initial hypotheses and/or questions you have of the data, ideas
- Data dictionary
- Instructions or an explanation of how someone else can reproduce your project and findings (What would someone need to be able to recreate your project on their own?)
- Key findings, recommendations, and takeaways from your project.

2. Final Report (.ipynb)

3. Acquire & Prepare Modules (.py)


4. Predictions (.csv).


5. non-final Notebook(s) (.ipynb)



### Nice to haves (With more time):
More time.  There is a lot of data to clean and analyze.  This is a much bigger project in the real world than what we can do in three days.  Principal among this is to correct for the dependence of the features that we put into the Regresssion models.  We are technically not supposed to have highly dependent features in our models and should either pick the best or do some feature engineering on these to create one feature that we can put into the model.   With more time I would do this as well as explore other features that are independent of the features chosen to see if they might add to future models.  I would also perhaps try to divide the train validate and test after grouping by zip code and running a linear regression on each zip code seperately.   


***

## <a name="findings"></a>Key Findings:
[[Back to top](#top)]
The key findings are that the square footage, the number of bathrooms, and the number of bedrooms are highly correlated with the tax assessed value but they are also highly correlated with each other which may be affecting the ability of the model to do particularly well.  Though they are doing better than the baseline, the RMSE values are till not in any range that can be applied in the real world.  With the average home value around 527,143 and a median around 409,260, having an RMSE of around 487,522 renders this model practically meaningless.  Much more work and exploration are required to build a model which would be helpful

***

## <a name="dictionary"></a>Data Dictionary  
[[Back to top](#top)]
bedrooms	bathrooms	sqft	tax_value	year_built	tax_amount	Los Angeles	Orange	Ventura
### Data Used
---
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| bedrooms | The number of bedrooms in the home in question | float64 |
| bathrooms | The number of bathrooms in the home in question | float64 |
| sqft | The square footage of the home | float64 |
| tax_value | The assessed tax value of the home | float64 |
| Los Angeles | Whether or not the home is in Los Angeles | uint8 |
| Orange | Whether or not the home is in Orange | uint8 |
| Ventura | Whether or not the home is in Ventura | uint8 |

***

## <a name="wrangle"></a>Data Acquisition and Preparation
[[Back to top](#top)]
Basically I fumbled around until I figured it out little by litte.
![]()


### Wrangle steps: 


*********************

## <a name="explore"></a>Data Exploration:
[[Back to top](#top)]
- Python files used for exploration:
    - wrangle.ipynb
    - explore.ipynb
    - modeling.ipynb
    - classification_exercises.ipynb
    - sub_final_report_with_independent_discovery.ipynb


### Takeaways from exploration:

Main drivers are month-to-month contract, Fiber Optic customers and tenure/charge ratio
***

## <a name="stats"></a>Statistical Analysis
[[Back to top](#top)]

### Stats Test 1: Chi Squared Test

The main statistical test used was the Chi Squared Test to determine dependence between the target variable churn_Yes and the features.  A for loop was used to run the Chi Squared test for each column that was a feature as compared to the target.  

#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is that the target column (Churn_Yes) is independent of the column in question
- The alternate hypothesis (H<sub>1</sub>) is that the target column (Churn_Yes) is dependent on the column in question

#### Confidence level and alpha value:
- For the purposes of these tests, I used an alpha < 0.01 since the number of customers was high and there were so many features to choose from.

#### Results:
After removing the redundant columns in the original dataframe; "Unnamed: 0", "gender_male", "phone_service_Yes", "multiple_lines_No phone service" all did not meet the p-value we set and thus we didn't have sufficient evidence to reject the null hypothesis. I therefore dropped them all from the analysis. 

"Unnamed: 0" is just an id number so it makes sense that they are independant events...kinda goes without saying but a good sanity check.

#### Summary:
This left us with 22 features and one target to work with which are as follows:
'senior_citizen', 
'partner_Yes', 
'dependents_Yes', 
'multiple_lines_Yes',
'online_security_Yes', 
'online_backup_Yes', 
'device_protection_Yes',
'tech_support_Yes', 
'streaming_tv_Yes', 
'streaming_movies_Yes',
'paperless_billing_Yes',
'internet_service_type_Fiber optic', 
'internet_service_type_None',
'payment_type_Credit card (automatic)', 
'payment_type_Electronic check',
'payment_type_Mailed check', 
'contract_type_Month-to-month',
'contract_type_One year', 
'contract_type_Two year', 
'tenure_normalized',
'monthly_charges_normalized', 
'churn_Yes'

### Stats Test 2: Pearson Correlation
- Looking at the scatterplot of the monthly charges vs tenure hued on churn, I decided that the tenure to charge ratio was an interesting new column I could make since it seemed new customers who pay a lot of money churn.  
- I ran the Pearsonr Correlation test on the monthly charges and tenure in order to see how strong the correlation was
#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is  that there was no correlation
- The alternate hypothesis (H<sub>1</sub>) is  that there is some correlation
- An alpha < 0.05 was chosen
    - An r value of 0.401285600132126 indicated a moderate correlation while
    - The p-value of 3.0121978116585465e-73 showed a significant result
    
Thus I felt confident making the new column 'tenure_charge_ratio' as an added feature.  I also thought that this correlation might be stronger under a non-linear correlation test so I moved on to Spearman's Correlation


### Stats Test 3: Spearman Correlation
- It was quite obvious that the regression line would be better fit if it were not linear.  So I ran a Spearman's Correlation test to make sure.  A stronger correlation with a curve bent toward the newest customers paying the most would present more evidence that using the newly made column would prove fruitful.


- I ran the Spearman's Correlation test on the monthly charges and tenure in order to see how strong the correlation was
#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is  that there was no correlation
- The alternate hypothesis (H<sub>1</sub>) is  that there is some (monotonically incresasing or decreasing) correlation
- Again an alpha < 0.05 was chosen
    - An r value of 0.4843751919531249 indicated a moderate but stronger correlation while
    - The p-value of 1.5136016239109236e-110 showed a significant result
-  These two tests combined convinced me that I was on the right track so I continued all models with this new column as an added feature
    


#### Results:
All of the columns seemed highly correlated to the target variable.  However so many of the columns seemed to be confounding features to the main two drivers of the entire thing.  It is clear that the more that the customers pay per month and the shorter timeframe that they have been customers is a main driving force behind churn.  Secondary to this and also slightly confounding is whether or not the customers were on a Month-to-Month contract, which in terms of the original features had the most significant result upon the chi squared test.  Furthermore, Fiber Optic Customers seem to churn at quite a high rate.  This could be a result of presumably a high price and its not worth it for them or there might be something wrong with the service.


#### Summary:
That said it is difficult to rigorously rule out the confounding variables with the tools we have used up to this point in the course.  As such I am forced to rely on intuition to guess what the main drivers are rather than prove.  The fact that the two that I have chosen are of high priority in the decision trees that I looked at as well as the fact that many of the columns seem to naturally clump together like having dependents and having multiple services and lines and thus being on a long term conract seem a natural consequence of the interrelated nature of these columns.  

Given more time I would try to seperate out these confounding features more and see if I could get a more comprehensive picture on how these features clump together.  I did try a nested for loop to see if any columns were independent of others using the chi squared test.  That double loop was in my original exploration work and I used an alpha of .75.  This is not a statistical result per se but I just used a high alpha to see if I could extract out the columns that seem to be independent of any of the other columns.  I was then left with a set of columns that are totally dependent on all other columns and another set of columns that have less dependence on at least one other column.   Such an inquiry came up empty as there were time considerations, but I would, if given more time explore this further.  

Also another driver seemed to be fiber optic.  There seemed to be a high proportion of fiber optic customers that were dissatisfied.  Interestingly enough the fiber optic ended up in the independent column which seems to indicate that it is a standalone feature that causes dissatisfaction.  There seeems to be less confounding factors with this feature.  Perhaps a deeper analysis into this feature will lead to stronger conclusions as to what is going on there.

***

***

## <a name="conclusion"></a>Conclusion:
I admittedly did not set goals properly when starting this project and just jumped right in.  I went all over the place but eventually came upon the proper answers.  The two main drivers of churn in terms of features were whether the customer had a month-to-month contract and whether the customer had fiber optic service.  These two alone seem to explain sixty two percent of the churning customers.  However qualitatively the newer the customer and the more they are paying per month, the higher the churn rate while longer term customers paying less seem to stay with the company.

There are many confounding features that make this analysis hard but I believe I have found the foundation of the mass of churning customers.  The remaining thirty eight percent of the customers can be analized in a further study.

As for the drivers that I found: Presumably the month-to-month contract is designed to entice new customers and thus will naturally have a higher churn rate.  Further inquiry into the phenomenon might provide insight into how to keep these customers i.e. giving them a discount or combining services in order to keep them.  However, there is something going on with Fiber Optic Service.  Perhaps it is expensive and customers don't feel it is worth the cost, or perhaps it is unreliable.  More information is needed in order to address these questions.
[[Back to top](#top)]