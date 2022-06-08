import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder, StandardScaler
matplotlib.use('QtAgg')

from statsmodels.stats.weightstats import CompareMeans
import statsmodels.formula.api as smf
import missingno as msno
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from collections import defaultdict

if __name__ == '__main__':
    recipients = pd.read_csv('recipients.csv')
    attempts = pd.read_csv('survey_attempts.csv')
    merged = pd.merge(recipients, attempts, on='recipient_id',
                      how='left', indicator=True)
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

    merged.loc[merged.recipient_id.isin(one_success) &
               merged.recipient_id.isin(recipient_noABC),
               'stage'] = 'Ineligible'

    # Find Review status
    recipient_yesABC = merged.recipient_id[inABC]
    notActive = merged.account_status == 'Not Active'
    recipient_notActive = merged.recipient_id[notActive]

    merged.loc[merged.recipient_id.isin(one_success) &
               merged.recipient_id.isin(recipient_yesABC) &
               merged.recipient_id.isin(recipient_notActive),
               'stage'] = 'Review'

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
    # To deal with these problematic cases, I replace them with missing values to be
    # imputed later.

    merged.age[merged.age == 9999] = pd.np.nan 

    merged.time_county[merged.time_county < 0] = pd.np.nan

    # merged = merged[merged['age'] <= 200]
    # merged = merged[merged['time_county'] > 0]

    merged.isna().sum()  # Check with field teams or relevant departments to resolve these issues

    # TODO add missingno package
    # There are some missing values in 

    msno.matrix(merged)
    plt.tight_layout()
    plt.show()
    plt.clf()

    # We can see there is a correlation between account_number and account_status being missing,
    # This would make sense considering that someone without an account number wouldn't have 
    # a status and vice versa. There is a perfect association between age and county missingness.

    msno.heatmap(merged, cmap='rainbow')
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()

    # To investigate this, let's look at recipients with a missing age
    county_missing = merged.loc[merged.county.isna()]
    age_missing = merged.loc[merged.age.isna()]
    county_missing.equals(age_missing)

    # So age is missing exactly when county is missing. There are two 
    # recipients that did not fill in these questions for multiple survey attempts. 
    # They may have been uncomfortable sharing that information.
    county_missing.recipient_id.nunique()

    # Impute the missing values
    cat_cols_na = ['account_status', 'county']
    num_cols_na = ['age', 'time_county']

    merged[cat_cols_na] = merged[cat_cols_na].astype('category')
    d_na = {col: {n: cat for n, cat in enumerate(merged[col].cat.categories)}
            for col in cat_cols_na}
    merged[cat_cols_na] = pd.DataFrame(
        {col: merged[col].cat.codes for col in cat_cols_na},
        index=merged.index
    )

    imp_num = IterativeImputer(estimator=ExtraTreesRegressor(),
                               initial_strategy='median',
                               max_iter=10, random_state=0)
    imp_cat = IterativeImputer(estimator=ExtraTreesClassifier(),
                               initial_strategy='most_frequent',
                               max_iter=10, random_state=0, missing_values=-1)
    # merged.county = imp.fit_transform(merged.county.values.reshape(-1, 1))
    merged[num_cols_na] = imp_num.fit_transform(merged[num_cols_na])
    merged[cat_cols_na] = imp_cat.fit_transform(merged[cat_cols_na])
    # Now retrieve the original labels.
    for col in cat_cols_na:
        merged[col].replace(d_na[col], inplace=True)

    # merged[['county', 'account_status']] = imp.fit_transform(merged[['county', 'account_status']])

    # merged.age = merged.groupby('county')['age'].transform(
    #     lambda x: x.fillna(x.median()))

    # merged.time_county = merged.groupby('county')['time_county'].transform(
    #     lambda x: x.fillna(x.median())
    # )
    
    # 2 Who to focus on?
    # By construction, the starting group has no successful surveys. The 
    # review group has at least one successful survey per person. 
    # It would be good to know what the chance of someone moving from
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
    plt.close()
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

    # Compute feature importance
    # https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
    reg = LogisticRegressionCV()
    cat_cols = ['month', 'account_status', 'county']
    num_cols = ['age', 'time_county']
    d = defaultdict(LabelEncoder)
    le_fit = merged[cat_cols].apply(lambda x: d[x.name].fit_transform(x))

    # Add a StandardScaler before computing feature importance
    scaler = StandardScaler()

    X = pd.concat((merged[num_cols], le_fit), axis=1)
    y = merged['one_success']

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    reg.fit(X_train, y_train)
    # print("Best alpha using built-in LogisticRegCV: %f" % reg.alpha_)
    print("Best score using built-in LogisticRegCV: %f" % reg.score(X_test, y_test))
    coef = pd.Series(reg.coef_.flatten(), index=X.columns)
    imp_coef = coef.sort_values()
    # import matplotlib
    # matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind="barh")
    plt.title("Feature importance using Logistic Regression Model")
    plt.tight_layout()
    plt.show()

    # It appears that the account_status and age have the largest effect on whether a survey is successful.
    # Age has a negative effect however. Recall that account_status and age are correlated. 
    merged.groupby('account_status')['age'].mean()
    # The Not active accounts are slightly older.
    # Run a test for the equality of means. The null hypothesis is that the means are equal.
    active_age = merged.age[merged.account_status == 'Active']
    notactive_age = merged.age[merged.account_status == 'Not Active']
    CompareMeans.from_data(active_age, notactive_age).ttest_ind()
    # The p-value is close to zero, rejecting the null hypothesis that the means of the two groups are equal.

    # Let's compare these results with those from a Tree based model
    reg_extra_tree = ExtraTreesClassifier(n_estimators=10)
    reg_extra_tree.fit(X_train, y_train)
    feat_imp = pd.Series(
        reg_extra_tree.feature_importances_,
        index=X.columns
    ).sort_values()
    print(f"Mean accuracy on test data is {reg_extra_tree.score(X_test, y_test)}")
    feat_imp.plot(kind="barh")
    plt.title("Feature importance using Extra Tree Classifier")
    plt.tight_layout()
    plt.show()
    # This time the model has age as the most important effect. All the effects are chosen by the
    # model, however. Note that there is randomness in the model, so sometimes some features
    # will be ranked differently. But the model will still usually choose all of the features for
    # prediction.