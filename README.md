# <a name="top"></a>ZILLOW DATASET 
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

## <a name="findings"></a>Key Findings, Recommendations and Takeaways:
[[Back to top](#top)]
The key findings are that the square footage, the number of bathrooms, and the number of bedrooms are highly correlated with the tax assessed value but they are also highly correlated with each other which may be affecting the ability of the model to do particularly well.  Though they are doing better than the baseline, the RMSE values are till not in any range that can be applied in the real world.  With the average home value around 527,143 and a median around 409,260, having an RMSE of around 487,522 renders this model practically meaningless.  Much more work and exploration are required to build a model which would be helpful.  The main takeaway is still that I need to first focus on making an MVP fefore trying to be perfect.

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
    - project_prelim.ipynb
    - acquire.py
    - prepare.py
    - wrangle.py
    - explore.py

The steps to look through the MVP are in the final notebook.  There are a lot of functions in the preliminary  exploration, the acquire and the prepare files that one can use to explore further but for the purposes of reproducing this mvp all that is needed is in the wrangle.py file and the project_final.ipynb.
### Takeaways from exploration:

Main drivers are number of bedrooms, the number of bathrooms, the square footage, year built, and the county the home is in
***

## <a name="Conclusion"></a>Conclusion:
[[Back to top](#top)]
This is a long way away from a viable real world product.  However it is a very good start.  