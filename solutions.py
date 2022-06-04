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

    # Calculate the 'stage' variable
    temp = merged.set_index('recipient_id')
    mask = merged.groupby('recipient_id')['success'].any()
    # Set the value to start for those with no successful surveys. '~' negates the value 
    # of the mask. In this case, ~mask means find those without any successful survey
    temp['stage'] = ''    
    temp.loc[~mask, 'stage'] = 'Start'
    temp = temp.reset_index()
    
    # Find Ineligible respondents
    one_success = [mask.index[i] for i, m in enumerate(mask) if m]
    inABC = temp.county.isin(['County A', 'County B', 'County C'])
    recipient_noABC = temp.recipient_id[~inABC]

    temp.loc[temp.recipient_id.isin(one_success) & temp.recipient_id.isin(recipient_noABC), 'stage'] = 'Ineligible'

    # Find Review status
    recipient_yesABC = temp.recipient_id[inABC]
    notActive = temp.account_status == 'Not Active'
    recipient_notActive = temp.recipient_id[notActive]

    temp.loc[temp.recipient_id.isin(one_success) & 
        temp.recipient_id.isin(recipient_yesABC) & 
        temp.recipient_id.isin(recipient_notActive), 'stage'] = 'Review'


    # Find Pay status
    active = temp.account_status == 'Active'
    recipient_active = temp.recipient_id[active]
    temp.loc[temp.recipient_id.isin(one_success) & 
        temp.recipient_id.isin(recipient_yesABC) & 
        temp.recipient_id.isin(recipient_active), 'stage'] = 'Pay'
    
    # There are 6 respondents without an account status. We could assign these to not Active
    # but it is important to follow up with a field team for this answer
    # For now we assign it to Nan
    temp.stage.replace('', pd.np.nan, inplace=True)


    # How many recipients in each stage?
    temp.stage.value_counts(dropna=False)

    # How many successful surveys in December
    temp['month'] = temp['date'].astype(str).str[:2]
    temp.groupby('month')['success'].sum() # 102 successful surveys in December

    # Abnormalities in the source data
    temp.describe()

    # minimum time in county is negative. Max age is 9999
    for i in temp.columns:
        print(temp[i].value_counts(dropna=False))
    # To deal with these problematic cases, I drop them from the data for analysis
    # Also drop the NaN
    temp = temp[temp['age'] <= 200]
    temp = temp[temp['time_county'] > 0]

    temp.isna().sum() # Check with field teams or relevant departments to resolve these issues

    # 2 Who to focus on?
    # Which percentage of recipients in the review stage have at least one successful survey
    # If this percentage is too low, it would be better to focus on the Start group. Over some
    # time you will collect enough data to know how likely they are to have at least one successful
    # survey. Then you can compare with the recipients in the review stage. Also consider the distance
    # to those in each stage. If the cost is too high for some reason, it will be better to focus on
    # the cheaper stage.