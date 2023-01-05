# Churn At Telco

# Project Description

The main goal of a business is to make money so that it can continue to stay operational and then (hopefully) go on to the second goal of helping people. In order to stay operational, businesses need to aquire and keep customers so that they may purhcases goods or services for the remainder of time. As we know, customers have many reasons for no longer buying a product or cutting ties with a service. We as Data Scientists are going to look into the reason WHY customers at Telco are deciding to churn. Once we get to the WHY, we can make recommendations on HOW we can provide solutions to help retain customers and prevent them from churning.

# Project Goal

- Discover drivers of churn at Telco
- Use drivers to develop a machine learning model that accurately predicts customer churn
- Churning is defined as a customer cutting ties with the service(s) Telco provides
- This information could be used to further our understanding of how customers at Telco (and simliar companies)     think and help other companies better understand consumer behavior 

# Initial Thoughts

- Are customers with Tech Support more or less likely to churn?
- What contract type churns the most?
- Does being a senior citizen have an affect on churn?
- Do customers who churn have paperless billing?


# The Plan

- Acquire data from Sequel Ace database
- Removed columns that did not contain useful information (payment_type_id, internet_service_type_id, etc.)
- Checked columns to promote readability (all were good)
- Checked for nulls in the data and dropped null values stored as white space (used .strip)
- Checked that column data types were appropriate and had to change total_charges to from object to a float
- Added additional features to investigate:
- Converted binary categorical variables to numeric (yes/no & male/female)
- Added dummy variables for the non-binary categorical variables and then concatenated them to the original   dataframe

#### Explore data in search of drivers of churn

##### Answer the following initial questions
- Are customers with Tech Support more or less likely to churn?
- What contract type churns the most?
- Does being a senior citizen have an affect on churn?
- Do customers who churn have paperless billing?

# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|gender|Binary, specifies if male or female|
|senior_citizen|True or False, specifies if senior citizen|
|partner|True or False, specifies if customer has a partner|
|dependents|True or False, specifies if customer has a dependent|
|tenure|Integer, how many months the person has been a customer with Telco|
|phone_service|Binary, Yes or No, specifies if customer has phone service|
|multiple_lines|Binary, Yes or No, specifies if customer has multiple lines|
|online_security|Binary, Yes or No, specifies if customer has online security|
|online_backup|Binary, Yes or No, specifies if customer has online backup|
|device_protection|Binary, Yes or No, specifies if customer has device protection|
|tech_support|Binary, Yes or No, specifies if customer has tech support|
|streaming_tv|Binary, Yes or No, specifies if customer streams tv|
|streaming_movies|Binary, Yes or No, specifies if customer streams movies|
|paperless_billing|Binary, Yes or No, specifies if customer has paperless billing|
|monthly_charges|Charges in a month, measured in dollars|
|total_charges|Total charges overal, measured in dollars|
|churn|The word used to define if a customer has left Telco|
|contract_type|Specifies contract type, month-to-month, two-year, one-year|
|payment_type|Specifies payment type, electronic check, mailed check, bank transfer, credit card|
|internet_service_type|Specifies internet service type, fiber optic, dsl, or none|

# Steps to Reproduce

- Clone this repo.
- Acquire the data from Kaggle
- Put the data in the file containing the cloned repo.
- Run notebook.

# Takeaways and Conclusions

- Customers with tech support are less likely to churn

- The biggest driver of churn appears to be contract type. Customers who have month-to-month contracts churn at a rate just over 40%

- Paperless billing does not appear to have a big affect on churn rate.

- Whether or not a customer has phone service doesn't play a significant role in churn.

- Age and gender do not appear to have a significant affect on churn.

- Monthly charges do appear to influence churn rate.


# Recommendations

- Creating a savings bundle (like Progressive) for customers who have multiple products can help incentivize using more services.
- Creating a savings bundle (like some insurance companies) for customers who have multiple products can help incentivize using more services.
- Use Tech Support as a priority and differentiator  for not only customer service, but for aquiring customer trust
- Give incentives to month-to-month contracts as it is the most popular contract type that churns. Incentives could include (for example) a temporarily free service such as Tech Support or if they were to go paperless with an app we could apply a discount while also collecting customer data to help with furture data science solutions. 
