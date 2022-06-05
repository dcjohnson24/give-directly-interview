import pandas as pd
import matplotlib.pyplot as plt


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
    
    inABC = merged.county.isin(['County A', 'County B', 'County C'])
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
    # TODO code here for number of recipients that transition
    (merged.groupby('recipient_id')['stage'].nunique() > 1).sum()

    # We can also look at geographical distances to determine transportation costs
    # For both the start and review stages, there are only 
    # 
    # Also the Start group is bigger with 148 recipients compared to only 38 in the Review group.
    # the start group success rate only needs to be 38/148 ~ 26% to match a 100% success rate in the
    # Review group. It would be good to check this with historical data on the conversion success rate
    merged.stage.value_counts(dropna=False)      
