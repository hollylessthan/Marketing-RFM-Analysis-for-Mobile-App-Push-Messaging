
# # Tuango - RFM Analysis for Mobile App Push Messaging
# 

import pandas as pd
import pyrsm as rsm
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


tuango = pd.read_pickle("data/tuango.pkl")
tuango.ordersize


# ## Part I:  Preliminary and Quintile Analysis (Q1 to Q6, 3 points each)
# 
# ### 1. What percentage of customers responded (i.e., bought anything) after the push message?


tuango.userid.nunique() == len(tuango) #test number of unique customers

rep_rate = len(tuango[tuango.buyer == 'yes']) / len(tuango) 
rep_rate #2.98%


# 
# ### 2. What was the average amount spent on the Karaoke deal by customers that bought one (or more)? 

avg_ordersize = tuango[tuango.buyer == 'yes'].ordersize.mean() 
avg_ordersize #202.12876


# ### 3.	Create independent quintile variables for recency, frequency and monetary. 

tuango = tuango.assign(rec_iq=rsm.xtile(tuango["recency"], 5))
tuango = tuango.assign(freq_iq=rsm.xtile(tuango["frequency"], 5, rev = True))
tuango = tuango.assign(mon_iq=rsm.xtile(tuango["monetary"], 5, rev = True))


# ### 4.	Create bar charts showing the response rate (i.e., the proportion of customers who bought something) for this deal per (independent) recency, frequency, and monetary quintile (i.e., 3 plots).


fig_r = rsm.prop_plot(tuango, "rec_iq", "buyer", "yes")
fig_r = fig_r.set(xlabel="Recency quintile")



help(rsm.prop_plot)



fig_f = rsm.prop_plot(tuango, "freq_iq", "buyer", "yes")
fig_f = fig_f.set(xlabel="Frequency quintile")




fig_m = rsm.prop_plot(tuango, "mon_iq", "buyer", "yes")
fig_m = fig_m.set(xlabel="Monetary quintile")


# ### 5.	Create bar charts showing the average amount spent (in RMB) (i.e., ordersize) per independent recency, frequency, and monetary quintile using only those customers who placed an order after the push message
# 

rec_df = tuango[tuango.buyer == 'yes'].groupby("rec_iq")["ordersize"].mean().reset_index()
sns.barplot(x="rec_iq", y="ordersize", data=rec_df)



fre_df = tuango[tuango.buyer == 'yes'].groupby("freq_iq")["ordersize"].mean().reset_index()
sns.barplot(x="freq_iq", y="ordersize", data=fre_df)




mon_df = tuango[tuango.buyer == 'yes'].groupby("mon_iq")["ordersize"].mean().reset_index()
sns.barplot(x="mon_iq", y="ordersize", data=mon_df)


# ### 6.	What do the above bar charts reveal about the likelihood of response and the size of the order across the different recency, frequency, and monetary quintiles?

# The likelihood of response decreases as the independent recency, frequency, and monetary quintiles increase from 1 to 5. However, the order size across the different recency, frequency, and monetary quintiles does not show much difference.

# ## Part II: Profitability Analysis 
# 

# ### 7. What is the breakeven response rate?


#Create two RFM indices

#rfm_iq using the independent quintile approach
tuango = tuango.assign(
    rfm_iq=tuango["rec_iq"].astype(str)
    + tuango["freq_iq"].astype(str)
    + tuango["mon_iq"].astype(str)
)
#rfm_sq using the sequential quintile approach
tuango = tuango.assign(
    freq_sq=tuango.groupby("rec_iq")["frequency"].transform(rsm.xtile, 5, rev=True)
)
tuango = tuango.assign(
    mon_sq=tuango.groupby(["rec_iq", "freq_sq"])["monetary"].transform(rsm.xtile, 5, rev=True)
)

tuango = tuango.assign(
    rfm_sq=tuango["rec_iq"].astype(str)
    + tuango["freq_sq"].astype(str)
    + tuango["mon_sq"].astype(str)
)


marginal_cost = 2.5 #for all deals

breakeven = marginal_cost / (0.5 * avg_ordersize)
breakeven


# ### 8. What is the projected profit in RMB and the return on marketing expenditures if you offer the deal to all 250,902 remaining customers?


n = 250902
profit_nt = rep_rate * n * (0.5 * avg_ordersize) - marginal_cost * n
ROME_nt = profit_nt / (marginal_cost * n)

print(
    f"""If the company offers the deal to all 250,902 remaining customers, the projected profit is ${float(profit_nt):,} and the ROME is {round((100 * ROME_nt), 2)}%"""
)


# ### 9. Evaluate the performance implications of offering the deal to only those customers (out of 250,902) in RFM cells with a response rate greater than the breakeven response rate. Generate your result based on both sequential and independent RFM. Determine the projected profit in RMB and the return on marketing expenditures for each approach.

def response_rate(x, lev="yes"):
    return np.nanmean(x == lev)
def mailto(x, lev="yes", breakeven=0):
    return np.nanmean(x == lev) > breakeven

def perf_calc(df, base):
    df = df.assign(
    rfm_iq_resp=df.groupby(base)["buyer"].transform(response_rate),
    mailto_iq=df.groupby(base)["buyer"].transform(mailto, breakeven=breakeven)
    )
    perc_mail = np.nanmean(df["mailto_iq"])
    #avg_ordersize = tuango[tuango.buyer == 'yes'].ordersize.mean() 
    ordersize_mail = df[(df.buyer == 'yes')].ordersize.mean() 
    nr_mail = n * perc_mail
    rep_rate = np.nanmean(df.loc[df["mailto_iq"], "buyer"] == "yes")
    nr_resp = nr_mail * rep_rate
    mail_cost = marginal_cost * nr_mail
    profit = (0.5 * ordersize_mail) * nr_resp - mail_cost
    ROME = profit / mail_cost
    return (profit, ROME)
    #return(ordersize_mail)




perf_calc(tuango, "rfm_iq")



# calculate performance for independent RFM
profit_iq, ROME_iq = perf_calc(tuango, "rfm_iq")

print(
    f"""Based on independent RFM, the projected profit is ${float(profit_iq):,} and the ROME is {round((100 * ROME_iq), 2)}%"""
)


# calculate performance for sequential RFM
profit_sq, ROME_sq = perf_calc(tuango, "rfm_sq")

print(
    f"""Based on sequential RFM, the projected profit is ${float(profit_sq):,} and the ROME is {round((100 * ROME_sq), 2)}%"""
)


# ### 10.	What do you notice when you compare the `rfm_iq` and `rfm_sq` variables? Do the two approaches generally yield the same RFM index for any given customer? What do you see as the pros and cons of the two approaches (from a statistical as well as logical perspective) and why?

print(len(tuango[tuango.rfm_iq == tuango.rfm_sq])) #17594
print(len(tuango[tuango.rfm_iq != tuango.rfm_sq])) #10284

plt.figure(figsize=(16, 7))
fig = rsm.prop_plot(tuango, "rfm_iq", "buyer", "yes")
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
fig = fig.set(xlabel="Independent RFM index (rsm_iq)")



plt.figure(figsize=(16, 7))
fig = rsm.prop_plot(tuango, "rfm_sq", "buyer", "yes")
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
fig = fig.set(xlabel="Sequential RFM index (rsm_sq)")


# The two approaches yield the same RFM index for 17594 customers but different RFM indexes for 10284 customers.
# 
# The independent RFM makes the interpretation of each of the three RFM components easily understood. For instance, a frequency score of 5 for one customer with a recency rank of 5 means the same as a frequency score of 5 for another customer with a recency rank of 3. However, for smaller samples, this method will probably generate a less even distribution of combined RFM scores.
# 
# The sequential RFM provides a more even distribution of RFM scores than the independent RFM. However, its frequency and monetary rank scores are more difficult to interpret than those of the independent RFM. For example, a frequency rank of 5 for a customer with a recency rank of 5 may not mean the same thing as a frequency rank of 5 for a customer with a recency rank of 4 because the frequency rank is dependent on the recency rank.

# ### 11.	The answer to question 9 assumes a single breakeven response rate that applies across all cells. Redo your analysis for sequential RFM based on a breakeven response rate calculated for each RFM cell.
# 
# > Note: To project the expected profits for the remaining customers not part of the sample you can use Tuango’s standard fee and the ordersize number you calculated for question 2.

def perf_calc_rfm_breakeven(df, base):
    avg_orsz = df[df.buyer == 'yes'].groupby(base).ordersize.mean().reset_index().rename(columns={'ordersize':'avg_ordersize'})
    avg_orsz["breakeven"] = marginal_cost / (0.5 * avg_orsz["avg_ordersize"]) 
    df = df.merge(avg_orsz, how = "left", on = "rfm_sq")
    
    df = df.assign(
    rfm_sq_resp=df.groupby(base)["buyer"].transform(response_rate)
    )
    df["mailto_sq"]=df["rfm_sq_resp"] > df["breakeven"]
    
    
    perc_mail = np.nanmean(df["mailto_sq"])
    nr_mail = n * perc_mail
    rep_rate = np.nanmean(df.loc[df["mailto_sq"], "buyer"] == "yes")
    nr_resp = nr_mail * rep_rate
    mail_cost = marginal_cost * nr_mail
    ordersize_mail = df[(df["mailto_sq"] == True) & (df.buyer == 'yes')].ordersize.mean() 
    profit = (0.5 * ordersize_mail) * nr_resp - mail_cost
    ROME = profit / mail_cost
    return ((profit, ROME))



profit_sq_ab, ROME_sq_ab = perf_calc_rfm_breakeven(tuango, "rfm_sq")

print(
    f"""Based on a breakeven response rate calculated for each RFM cell and the sequential RFM, the projected profit is ${float(profit_sq_ab):,} and the ROME is {round((100 * ROME_sq_ab), 2)}%"""
)


# ### 12.	The answer to question 9 does not account for the fact that the response rate for each cell is an estimated quantity (i.e., it has a standard error). Redo your analysis for both independent and sequential RFM, adjusting for the standard error of the response rate in each cell. What implications can you draw from the difference in predicted performance compared to question 9?

# insert calculation code here
tuango["buyer_yes"] = np.where(tuango["buyer"] == "yes", 1, 0)

def mailto_lb(x, lev="yes", breakeven=0):
    x = x == lev
    return (np.nanmean(x) - 1.64 * rsm.seprop(x)) > breakeven

def perf_calc_lb(df, base):
    df = df.assign(
    mailto_lb=df.groupby(base)["buyer"].transform(mailto_lb, breakeven=breakeven)
    )
    perc_mail = np.nanmean(df["mailto_lb"])
    ordersize_mail = df[(df["mailto_lb"] == True) & (df.buyer == 'yes')].ordersize.mean() 
    nr_mail = n * perc_mail
    rep_rate = np.nanmean(df.loc[df["mailto_lb"], "buyer"] == "yes")
    nr_resp = nr_mail * rep_rate
    mail_cost = marginal_cost * nr_mail
    profit = (0.5 * ordersize_mail) * nr_resp - mail_cost
    ROME = profit / mail_cost
    return (profit, ROME)

#tuango["mailto_lbiq"] = tuango.groupby("rfm_iq")["buyer"].transform(mailto_lb, breakeven=breakeven)
#tuango["mailto_lbsq"] = tuango.groupby("rfm_sq")["buyer"].transform(mailto_lb, breakeven=breakeven)




profit_lbiq, ROME_lbiq = perf_calc_lb(tuango, "rfm_iq")
profit_lbsq, ROME_lbsq = perf_calc_lb(tuango, "rfm_sq")


print(
    f"""After adjusting the standard error of the response rate in each cell and based on the independent RFM, the projected profit is ${float(profit_lbiq):,} and the ROME is {round((100 * ROME_lbiq), 2)}%"""
)
print(
    f"""After adjusting the standard error of the response rate in each cell and based on the sequential RFM, the projected profit is ${float(profit_lbsq):,} and the ROME is {round((100 * ROME_lbsq), 2)}%"""
)


# The expected profits for both independent and sequential RFM based on the adjusted response rates are lower than those in the questions 9. However, the ROMEs for both independent and sequential RFM based on the adjusted response rates are much higher than those in the questions 9.

# ### 13.	Create a bar chart with profit information and a bar chart with ROME numbers for the analyses conducted in questions 9, 11, and 12


# Updating profit and ROME plots:
dat = pd.DataFrame(
    {
        "name": [
            "No targeting",
            "Indep. RFM",
            "Seq. RFM",
            "Seq. breakeven RFM",
            "Indep. lb RFM",
            "Seq. lb RFM"
        ],
        "Profit": [
            profit_nt,
            profit_iq,
            profit_sq,
            profit_sq_ab,
            profit_lbiq,
            profit_lbsq
        ],
        "ROME": [ROME_nt, ROME_iq, ROME_sq, ROME_sq_ab, ROME_lbiq, ROME_lbsq],
    }
)
plt.figure(figsize=(6, 4))
fig = sns.barplot(x="name", y="Profit", color="slateblue", data=dat)
fig.set(xlabel="", ylabel="Profit", title="Campaign profit")
fig.set_xticklabels(fig.get_xticklabels(),rotation=60)
for index, row in dat.iterrows():
    fig.text(
        row.name, row.Profit - 6000, f"{int(row.Profit):,}", ha="center", color="black"
    )




plt.figure(figsize=(6, 4))
fig = sns.barplot(x="name", y="ROME", color="slateblue", data=dat)
fig.set(xlabel="", ylabel="ROME", title="Return on Marketing Expenditures (ROME)")
fig.set_xticklabels(fig.get_xticklabels(),rotation=60)
for index, row in dat.iterrows():
    fig.text(
        row.name,
        row.ROME - 0.06,
        f"{round((100*row.ROME), 2):,}%",
        ha="center",
        color="white",
    )


# ### Question 14
# 
# Make a copy of the current notebook `tuango.ipynb` and name the copy `tuango_post.ipynb`. You also have access to a dataset with the results from the SMS roll-out (tuango_post.pkl). Tuango actually contacted all 250,902 customers. The data has a "training" variable (training = 1 for the data used in the test, training = 0 for the remaining customers). You can use this variable to help evaluate the actual performance for each of the different RFM approaches. Re-create the plot in question 13 based on this new dataset. 
# 
# Instead of using the “tuango.pkl” data, use “tuango_post.pkl”. 
# 
# Hint: It is important to note that, you should not use any information about buyers that were not in the test sample (i.e., training == 0) to calculate the break-even response rate etc. for targeting. 
