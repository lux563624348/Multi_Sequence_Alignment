from optparse import OptionParser
from datetime import datetime, date
import sys, time, multiprocessing
import pandas as pd

def date_to_daystamp(date_obj):
  datetime_obj = datetime(year=date_obj.year, month=date_obj.month, day=date_obj.day)
  timestamp = datetime_obj.timestamp()/(24*60*60) # in days
  return timestamp

def Levenshtein_Distance_with_Transposition_Date(seq1, seq2, dates1, dates2, dict_sub_matrix, max_transposition_date):
  """
  Calculates the Levenshtein distance between two sequences, considering transpositions.
  Args:
    seq1: The first sequence.
    seq2: The second sequence.
    dict_sub_matrix: Pandas DataFrame containing substitution distance.
  Returns:
    The Levenshtein distance between the two sequences.
  """
  seq1 = "#"+''.join(seq1)
  seq2 = "#"+''.join(seq2)
  #dates1 = np.insert(dates1, 0,  "None")
  #dates2 = np.insert(dates2, 0,  "None")  # Adding None to align dates with the prefixed '#'
  dates1 = ["None"] + [x for x in dates1]
  dates2 = ["None"] + [x for x in dates2]
  #print (dates1, dates2)
  m = len(seq1)
  n = len(seq2)
  # Create a distance matrix
  #dp = np.full((m, n), 0)  # Initialize with infinity to handle transpositions
  dp = [[0 for col in range(n)] for row in range(m)]
  # Initialize the first row and column
  for i in range(1, m):
    dp[i][0] = i
  for j in range(1, n):
    dp[0][j] = j

  # Fill the DP table
  for i in range(1, m):
    for j in range(1, n):
      # Standard costs
      insertion_cost = dp[i-1][j] + dict_sub_matrix[seq1[i-1]][seq2[0]]  # (letter -> # )
      deletion_cost = dp[i][j-1] +  dict_sub_matrix[seq1[0]][seq2[j-1]]  # ( # -> letter)
      # Substitution cost
      cost = (0 if seq1[i] == seq2[j] else dict_sub_matrix[seq1[i]][seq2[j]])
      substitution_cost = dp[i-1][j-1] + cost
      dp[i][j] = min(insertion_cost, deletion_cost, substitution_cost)
      # Handle transpositions with date constraint
      if ((i > 0) & (j > 0)):
          for idx_seq1  in range(1, m, 1):
              date_diff1 = abs(date_to_daystamp(dates1[idx_seq1]) - date_to_daystamp(dates2[j]))
              if (date_diff1 < max_transposition_date):
                  cost = (0 if seq1[idx_seq1] == seq2[j] else dict_sub_matrix[seq1[idx_seq1]][seq2[j]])
                  transposition_cost = dp[i-1][j-1] + cost
                  dp[i][j] = min(dp[i][j], transposition_cost)
          for idx_seq2  in range(1, n, 1):
              date_diff2 = abs(date_to_daystamp(dates1[i]) - date_to_daystamp(dates2[idx_seq2]))
              if (date_diff2 < max_transposition_date):
                  cost = (0 if seq1[i] == seq2[idx_seq2] else dict_sub_matrix[seq1[i]][seq2[idx_seq2]])
                  transposition_cost = dp[i-1][j-1] + cost
                  dp[i][j] = min(dp[i][j], transposition_cost)

      #print (dp)
      #break
    #break
  # Return the distance and matrix (optional)
  return dp[m - 1][n - 1], dp

def Normalize_Levenshtein_Distance_Score(seq1, seq2, dates1, dates2, dict_sub_matrix, max_transposition_date):
    ''' 0~1'''
    distance, matrix = Levenshtein_Distance_with_Transposition_Date(seq1, seq2, dates1, dates2, dict_sub_matrix, max_transposition_date)
    #print ("Leven_Distance", distance)

    seq1_match = [dict_sub_matrix[char][char] for char in seq1]
    seq2_match = [dict_sub_matrix[char][char] for char in seq2]

    seq1_penal = [dict_sub_matrix[char]["#"] for char in seq1] ## (letter -> # ) deletion
    seq2_penal = [dict_sub_matrix[char]["#"] for char in seq2]

    distance_max = max(sum(seq1_penal), sum(seq2_penal))
    distance_min = min(sum(seq1_match[::-1]), sum(seq2_match[::-1]))
    normalized_score = (distance - distance_min)/ (distance_max-distance_min)
    if (distance_max == distance_min): normalized_score = 1
    similarity_score = 1 - normalized_score
    return similarity_score

def Main_Compute_Similarity_For_Pair_and_Save(filename, pair_name, df_p1, df_p2, col_seq, col_time, dict_sub_matrix, max_transposition_date):
    df_seq1 = df_p1.sort_values(col_time)
    df_seq2 = df_p2.sort_values(col_time)

    seq1 = ''.join(df_seq1.loc[:, col_seq])
    seq2 = ''.join(df_seq2.loc[:, col_seq])
    date1 = df_seq1.loc[:, col_time].values
    date2 = df_seq2.loc[:, col_time].values

    dict_sub_matrix = dict_sub_matrix
    similarity = Normalize_Levenshtein_Distance_Score(seq1, seq2, date1, date2, dict_sub_matrix, max_transposition_date)

    with open(filename, 'a') as f:
        f.write(pair_name+'\t'+ str(similarity)+'\n')
        time.sleep(0.01)
    
    return None

def initializer(l, f):
    global lock, filename
    lock = l
    filename = f

def main():
    if len(sys.argv) != 5:
        #print (sys.argv)
        print("Usage: python MSA.py path_data, path_sub_matrix, max_transposition_date, core")
        sys.exit(1)

    path_data = sys.argv[1]
    path_sub_matrix = sys.argv[2]
    max_transposition_date = sys.argv[3]
    num_threads = int(sys.argv[4])
    
    Path_data = "/home/xli_p14/github/sentenceTransformers/demo_Input_File.tsv"
    Path_matrix = "/home/xli_p14/github/sentenceTransformers/test_submatrix.txt"
    max_transposition_date = 5
    
    col_ID = 'ID'
    col_time = 'date'
    col_seq = 'event'
    
    df_tem = pd.read_csv(Path_data, sep='\t')
    df_tem.loc[:, 'date'] = pd.to_datetime(df_tem.loc[:, col_time])
    dict_matrix = pd.read_csv(Path_matrix, sep='\t', index_col=0).to_dict()
    df_groups = df_tem.groupby(col_ID)
    IDs = list(df_groups.groups.keys())
    
    similarity_matrix = []
    start_time = time.time()
    
    num_cpu = multiprocessing.cpu_count()
    N_thread = min(len(df_groups), num_threads, num_cpu)
    ## If threads of server is not enough to run chrs at the same time. Choose CPU core as max.

    filename = "test_output.txt"
    #lock = multiprocessing.Lock()
    #pool = multiprocessing.Pool(N_thread, initializer=initializer, initargs=(lock, filename))
    
    pool = multiprocessing.Pool(N_thread)
    
# Initialize the pool with the lock and filename
    
    print ('Threads: '+ str(N_thread))
    
    for idx in range(len(IDs)):
        print ("Process ID: ", idx)
        for idj in range(idx):
            df_p1 = df_groups.get_group(IDs[idx]).loc[:, [col_time, col_seq]]
            df_p2 = df_groups.get_group(IDs[idj]).loc[:, [col_time, col_seq]]
            pair_name = IDs[idx] + "_" + IDs[idj]
            pool.apply_async(Main_Compute_Similarity_For_Pair_and_Save, args=(filename, pair_name, df_p1, df_p2, col_seq, col_time, dict_matrix, max_transposition_date))
            #Similarity = Main_Compute_Similarity_For_Pair(pair_name, df_p1, df_p2, col_seq, col_time, dict_matrix, max_transposition_date)
            #similarity_matrix.append([pair_name, Similarity])
    pool.close()
    pool.join()
    print('All subprocesses done.')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")
    return None
    
if __name__ == "__main__":
    main()