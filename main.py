# Imports #
from constants import *
import recsys as rs
from functions import *
####################

# Data imports #
df_occ_kitchen = postprocess_data(prepare_data(import_data(dataset_path_occ_kitchen)), start_date=start_date, end_date=end_date)
tv_df = postprocess_data(prepare_data(import_data(dataset_path_tv)), start_date=start_date, end_date=end_date)
# toaster_df = postprocess_data(prepare_data(import_data(dataset_path_toaster)), start_date=start_date, end_date=end_date)
# kettle_df = postprocess_data(prepare_data(import_data(dataset_path_kettle)), start_date=start_date, end_date=end_date)
####################

# Appliances #
tv = rs.Appliance(
    df=tv_df, 
    column='state', 
    amp_threshold=40, 
    width_threshold=10, 
    norm_amp=30,
    norm_freq=1,
    groupby='1d',
    df_occ=df_occ_kitchen)
# kettle = rs.Appliance(
#     df=kettle_df, 
#     column='state', 
#     amp_threshold=2500, 
#     width_threshold=20, 
#     norm_amp=3000,
#     norm_freq=3,
#     groupby='1d')
# toaster = rs.Appliance(
    # df=toaster_df, 
    # column='state', 
    # amp_threshold=500, 
    # width_threshold=20, 
    # norm_amp=600,
    # norm_freq=2,
    # groupby='1d')
####################

def main():
    # Instantiate a recommender
    rec_tv = rs.Recommender(app=tv)
    # rec_toaster = rs.Recommender(app=toaster)
    # rec_kettle = rs.Recommender(app=kettle)
    
    # Generate recommendations
    recs_tv = rec_tv.generate(freq=True, amp=True, occ=True)
    # recs_toaster = rec_toaster.generate(freq=True, amp=True, occ=False)
    # recs_kettle = rec_kettle.generate(freq=True, amp=True, occ=False)
    
    # Print recommendations
    # recs_explained = {
    #     [str(row.relevance) + " " + row.explanation for row in recs_tv],
    #     [str(row.relevance) + " " + row.explanation for row in recs_toaster],
    #     [str(row.relevance) + " " + row.explanation for row in recs_kettle],
    # }; print('TV:', *recs_explained, sep='\n')
    
    # Evaluate recommendations
    y_pred = rec_tv.y_pred;
    eval = rs.Evaluator(rec_tv, y_pred)
    print('rel:', len(eval.relevance))
    print('rel_recs:', len(eval.rel_recs))
    print(eval.report())
    eval.confusion_matrix()

main()