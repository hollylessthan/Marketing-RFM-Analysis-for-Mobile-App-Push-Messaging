# Marketing-RFM-Analysis-for-Mobile-App-Push-Messaging

This is a course project from the Customer Analytics course at UCSD, and the data set is obtained from the course as well. The goal of this project is to increase the performance of a mobile app push messaging campaign and maximize the generated profit.

#### Language
Python 3.8.5 (default, Jul 28 2020, 12:59:40)

#### Context
In this project, I used RFM analysis, a common marketing method, to evaluate and decide the targeting customers. RFM stands for recency, frequency, and monetary, and this method basicly analyze customers based on their recent purchase dates, the frequency of their purchases, and the amount spent on purchases. The specific steps for this project is outlined below:

1. calculated overall response rate and order size from customers.
2. calculated RFM quantile variables and compared the response rates and order sizes for each quantile variables.
3. based on the cost and profit assumption, calculated the break even response rate.
4. calculated and compared the projected profits and return on marketing expenditures if not targeting v.s. if targeting.
5. tested idependent breakeven rate for each quantile variables and adjusted standard errors for response rate
6. evaluated the actual performance

#### Result
Based on independent RFM and sequential RFM, the projected profit and ROME is significently greater than those if sending the message to all customers without targeting. Please see the .py file for more of the analysis result.
