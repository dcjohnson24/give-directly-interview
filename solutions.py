import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')

import statsmodels.formula.api as smf


if __name__ == '__main__':
    recipients = pd.read_csv('recipients.csv')
    attempts = pd.read_csv('survey_attempts.csv')
    merged = pd.merge(recipients, attempts, on='recipient_id', how='left', indicator=True)
    merged._merge.value_counts(dropna=False)

    # It appears that all the recipients have had some survey attempt. We can drop the _merge
    # variable now
    merged.drop(columns='_merge', inplace=True)
    # Check for duplicates. We expect that recipient_id and survey_id together form a unique id 
    merged[merged.duplicated(subset=['recipient_id', 'survey_id'], keep=False)]
    # There are duplicates that must be dropped.
    merged = merged.drop_duplicates(subset=['recipient_id', 'survey_id'], keep='first').reset_index(drop=True)

    # Calculate the 'stage' variable
    mask = merged.groupby('recipient_id')['success'].any()
    one_success = [mask.index[i] for i, m in enumerate(mask) if m]
    # # Set the value to start for those with no successful surveys. '~' negates the value 
    # # of the mask. In this case, ~mask means find those without any successful survey
    merged.loc[~merged.recipient_id.isin(one_success), 'stage'] = 'Start'  
    # Find Ineligible respondents
    
    # Remove the text 'County' from the column
    merged.county = merged.county.str.replace('County ', '', regex=False)
    inABC = merged.county.isin(['A', 'B', 'C'])
    recipient_noABC = merged.recipient_id[~inABC]

    merged.loc[merged.recipient_id.isin(one_success) & merged.recipient_id.isin(recipient_noABC), 'stage'] = 'Ineligible'

    # Find Review status
    recipient_yesABC = merged.recipient_id[inABC]
    notActive = merged.account_status == 'Not Active'
    recipient_notActive = merged.recipient_id[notActive]

    merged.loc[merged.recipient_id.isin(one_success) & 
        merged.recipient_id.isin(recipient_yesABC) & 
        merged.recipient_id.isin(recipient_notActive), 'stage'] = 'Review'


    # Find Pay status
    active = merged.account_status == 'Active'
    recipient_active = merged.recipient_id[active]
    merged.loc[merged.recipient_id.isin(one_success) & 
        merged.recipient_id.isin(recipient_yesABC) & 
        merged.recipient_id.isin(recipient_active), 'stage'] = 'Pay'
    
    # There are 6 respondents without an account status. We could assign these to not Active
    # but it is important to follow up with a field team for this answer.


    # How many recipients in each stage?
    merged.stage.value_counts(dropna=False)

    # How many successful surveys in December
    merged['month'] = merged['date'].astype(str).str[:2]
    merged.groupby('month')['success'].sum() # 102 successful surveys in December

    # Abnormalities in the source data
    merged.describe()

    # minimum time in county is negative. Max age is 9999
    for i in merged.columns:
        print(merged[i].value_counts(dropna=False))
    # To deal with these problematic cases, I drop them from the data for analysis
    # Also drop the NaN
    merged = merged[merged['age'] <= 200]
    merged = merged[merged['time_county'] > 0]

    merged.isna().sum() # Check with field teams or relevant departments to resolve these issues

    # 2 Who to focus on?
    # By construction, the starting group has no successful surveys. The review group has at least one
    # successful survey per person. It would be good to know what the chance of someone moving from
    # the review stage to a pay stage at a later date. 

    (merged.groupby('recipient_id')['stage'].nunique() > 1).sum()
    # There do not appear to be recipients that have changed their stage. Let's check for recipients
    # that have changed from Not Active to Active accounts

    (merged.groupby('recipient_id')['account_status'].nunique() > 1).sum()
    # No recipients that have changed from Not Active to Active accounts

    # Also the Start group is bigger with 148 recipients compared to only 38 in the Review group.
    # the Start group success rate only needs to be 38/148 ~ 26% to match a 100% success rate in the
    # Review group. It would be good to check this with historical data on the conversion success rate.
    # Data on the cost of converting the Start group vs the Review group would also be helpful
    # for the decision.
    merged.stage.value_counts(dropna=False)      

    # 3.1
    # Graph the relationship between age and at least one successful survey
    merged['one_success'] = 0
    merged.loc[merged.recipient_id.isin(one_success), 'one_success'] = 1
    # Create bins for age broken into quartiles
    merged['age_bin'] = pd.qcut(merged.age, q=4)
    age_success = merged.groupby('age_bin')['one_success'].sum()
    age = merged.groupby('age_bin')['one_success'].count()
    results = age_success.div(age, level='age_bin') * 100

    results.plot(kind='bar')

    plt.xticks(rotation=0)
    plt.xlabel('Age Group')
    plt.ylabel('% of recipients with at least\n one successful survey')
    plt.show()

    # TODO logistic regression of age and one_success
    logit = smf.logit('one_success ~ age', data=merged).fit()
    print(logit.summary())
    odds_ratios = pd.DataFrame({
            'OR': logit.params, 
            'Lower CI': logit.conf_int()[0],
            'Upper CI': logit.conf_int()[1]
        })
    
    odds_ratios = pd.np.exp(odds_ratios)

    odds_ratios['OR'][1] - 1
    # Each additional increase of one year in age is associated with a 18 percent decrease
    # in odds of having at least one successful survey.

    # It appears that older recipients are less likely to respond to surveys. Focusing on the 
    # older recipients would help target those with low response rates.

    # 3.2
    # We include other factors in our logistic regression to check for confounders.
    big_logit = smf.logit('one_success ~ time_county + age + month', data=merged).fit()
    print(big_logit.summary())
    # when we include time spent in the county and the month of the survey, no predictor is signficant

    another_logit = smf.logit('one_success ~ account_status', data=merged).fit()
    # account status is not a significant predictor.