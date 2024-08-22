## Function for calculate distance & similarity

def levenshtein_distance(seq1, seq2, dict_sub_matrix):
    """
    Calculates the Levenshtein distance between two sequences.
    Args:
    seq1: The first sequence (string).
    seq2: The second sequence (string).
    Returns:
    The Levenshtein distance between the two sequences.
    """
    seq1 = ["#"] +  [x for x in seq1]
    seq2 = ["#"] +  [x for x in seq2]
    m = len(seq1)
    n = len(seq2)
    # Ref: https://medium.com/@ethannam/understanding-the-levenshtein-distance-equation-for-beginners-c4285a5604f0
    # Create a distance matrix
    #dp = np.full((m, n), 0)
    dp = [[0 for col in range(n)] for row in range(m)]
    # Initialize the first row and column
    for i in range(1, m):
        dp[i][0] = i
    for j in range(1, n):
        dp[0][j] = j
    
    ## Fill the DP table
    for i in range(1, m):
        for j in range(1, n):
            #insertion_cost = dp[i-1][j] + df_sub_matrix.loc[seq1[i-1], seq2[0]]  ## ( # -> letter)
            #deletion_cost = dp[i][j-1] +  df_sub_matrix.loc[seq1[0], seq2[j-1]]  ## (letter -> # ) deletion
            #cost = (0 if seq1[i] == seq2[j] else df_sub_matrix.loc[seq1[i], seq2[j]])
            insertion_cost = dp[i-1][j] + dict_sub_matrix[seq1[i-1]][seq2[0]]
            deletion_cost  = dp[i][j-1] + dict_sub_matrix[seq1[0]][seq2[j-1]]
            cost = (0 if seq1[i] == seq2[j] else dict_sub_matrix[seq1[i]][seq2[j]])
            substitution_cost = dp[i-1][j-1] + cost # ## only +1 when i j not same
            dp[i][j] = min(insertion_cost, deletion_cost, substitution_cost)
        #break
    print ("Dis: ", dp[m - 1][n - 1])
    return dp

#date_format = '%Y-%m-%d'

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
  seq1 = ["#"] +  [x for x in seq1]
  seq2 = ["#"] +  [x for x in seq2]
  dates1 = ["None"] + [x for x in dates1]
  dates2 = ["None"] + [x for x in dates2]

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
  # for rareness-weight
  w1,w2,w3 = -0.5, 0.75, 0.5
    
  for i in range(1, m):
    for j in range(1, n):
      # Standard costs
      insertion_cost = dp[i-1][j] + dict_sub_matrix[seq1[i-1]][seq2[0]]  # (letter -> # )
      deletion_cost = dp[i][j-1] +  dict_sub_matrix[seq1[0]][seq2[j-1]]  # ( # -> letter)
      # Substitution cost
      cost = dict_sub_matrix[seq1[i]][seq2[j]] ## include for match
      substitution_cost = dp[i-1][j-1] + cost
      dp[i][j] = min(insertion_cost, deletion_cost, substitution_cost)
      # Handle transpositions with date constraint
      if ((i > 0) & (j > 0)):
          for idx_seq1  in range(1, m, 1):
              date_diff1 = abs((dates1[idx_seq1] - dates2[j]).days)
              if (date_diff1 < max_transposition_date):
                  sub_cost = dict_sub_matrix[seq1[idx_seq1]][seq2[j]] ## Vtns = Vmtc * w3  ([0,1]) w3 = 0.5
                  cost = (w3*sub_cost if seq1[idx_seq1] == seq2[j] else sub_cost)
                  transposition_cost = dp[i-1][j-1] + cost
                  dp[i][j] = min(dp[i][j], transposition_cost)
          for idx_seq2  in range(1, n, 1):
              date_diff2 = abs((dates1[i] - dates2[idx_seq2]).days)
              if (date_diff2 < max_transposition_date):
                  sub_cost = dict_sub_matrix[seq1[i]][seq2[idx_seq2]] ## Vtns = Vmtc * w3  ([0,1]) w3 = 0.5
                  cost = (w3*sub_cost if seq1[idx_seq1] == seq2[j] else sub_cost)                  
                  transposition_cost = dp[i-1][j-1] + cost
                  dp[i][j] = min(dp[i][j], transposition_cost)

  return dp[m - 1][n - 1], dp


def Levenshtein_Distance_with_Transposition_Date_Beta(seq1, seq2, dates1, dates2, dict_sub_matrix, max_transposition_date):
    """
    Calculates the Levenshtein distance between two sequences, considering transpositions.
    Args:
    seq1: The first sequence.
    seq2: The second sequence.
    dict_sub_matrix: Pandas DataFrame containing substitution distance.
    Returns:
    The Levenshtein distance between the two sequences.
    """
    seq1 = ["#"] +  [x for x in seq1]
    seq2 = ["#"] +  [x for x in seq2]
    dates1 = ["None"] + [x for x in dates1]
    dates2 = ["None"] + [x for x in dates2]
    
    sizerow = len(seq1)
    sizecol = len(seq2)
    # Create a distance matrix
    dp = [[0 for col in range(sizecol)] for row in range(sizerow)]
    w1,w2,w3 = -0.5, 0.75, 0.5
    
    # Iniate the DP table, index & column is the INDEL_cost + tns_cost
    
    for i in range(1, sizerow):
        indel_cost = dict_sub_matrix[seq1[i]]["#"]
        tns_cost = 0
        for j in range(1, sizecol):
            if (seq1[i] == seq2[j]): ## if sequence same check if within transpotion date
                date_diff = abs((dates1[i] - dates2[j]).days)
                if (date_diff < max_transposition_date): ## Vtns = Vmtc * w3  ([0,1]) w3 = 0.5
                    tns_cost = w3*dict_sub_matrix[seq1[i]][seq2[j]]
                    break ## as long as one transpotion found, exit searching.
                break
        dp[i][0] = dp[i-1][0] + indel_cost + tns_cost
    
    for j in range(1, sizecol):
        indel_cost = dict_sub_matrix["#"][seq2[j]]
        tns_cost = 0
        for i in range(1, sizerow):
            if (seq1[i] == seq2[j]): ## if sequence same check if within transpotion date
                date_diff = abs((dates1[i] - dates2[j]).days)
                if (date_diff < max_transposition_date): ## Vtns = Vmtc * w3  ([0,1]) w3 = 0.5
                    tns_cost = w3*dict_sub_matrix[seq2[j]][seq1[i]]
                    break ## as long as one transpotion found, exit searching.
                break
        dp[0][j] = dp[0][j-1] + indel_cost + tns_cost
    
    for i in range(1, sizerow):
        for j in range(1, sizecol):
          # Standard costs
          insertion_cost = dp[i][j-1] + dict_sub_matrix[seq1[0]][seq2[j]]  # ( # -> letter)
          deletion_cost = dp[i-1][j] + dict_sub_matrix[seq1[i]][seq2[0]]  # (letter -> # )
          # Substitution cost
          min_dp = dp[i-1][j-1]
          sub_cost = min_dp + min(dict_sub_matrix[seq1[i]][seq2[j]], dict_sub_matrix[seq2[j]][seq1[i]])## include for match
          dp[i][j] = min(insertion_cost, deletion_cost, sub_cost)
    
    return dp[sizerow - 1][sizecol - 1], dp

def Levenshtein_Distance_with_Transposition_Date_Final(seq1, seq2, dates1, dates2, dict_sub_matrix, max_transposition_date):
    """
    Calculates the Levenshtein distance between two sequences, considering transpositions.
    Args:
    seq1: The first sequence.
    seq2: The second sequence.
    dict_sub_matrix: Pandas DataFrame containing substitution distance.
    Returns:
    The Levenshtein distance between the two sequences.
    """
    seq1 = ["#"] +  [x for x in seq1]
    seq2 = ["#"] +  [x for x in seq2]
    dates1 = ["None"] + [x for x in dates1]
    dates2 = ["None"] + [x for x in dates2]
    
    sizerow = len(seq1)
    sizecol = len(seq2)
    # Create a distance matrix
    dp = [[0 for col in range(sizecol)] for row in range(sizerow)]
    w1,w2,w3 = -0.5, 0.75, 0.5
    
    for i in range(1, sizerow):
        for j in range(1, sizecol):
            # Standard costs
            Vmtc_row = dict_sub_matrix[seq1[i]][seq1[i]]
            Vmtc_col = dict_sub_matrix[seq2[j]][seq2[j]]
            insertion_cost = dp[i][j-1] + w1*w2*min(Vmtc_row, Vmtc_col)  # ( # -> letter)
            deletion_cost = dp[i-1][j] + w1*w2*min(Vmtc_row, Vmtc_col)  # (letter -> # )
            # Substitution cost
            min_dp = min(dp[i][j-1], dp[i-1][j-1], dp[i-1][j])
            if (seq1[i] != seq2[j]):
                sub_cost = min_dp + w1*min(Vmtc_row, Vmtc_col) # sub
            if (seq1[i] == seq2[j]): ## sub or tns or mtc
                if (abs((dates1[i] - dates2[j]).days) > max_transposition_date):
                    ## sub: same sequence, but out of date.
                    sub_cost = min_dp + w1*min(Vmtc_row, Vmtc_col)
                else: ## within tns date,
                    sub_cost = min_dp + min(Vmtc_row, Vmtc_col)
                    if(i!=j):## tns
                        sub_cost = min_dp + w3*min(Vmtc_row, Vmtc_col)
            dp[i][j] = round(min(insertion_cost, deletion_cost, sub_cost), 2)
    return dp[sizerow - 1][sizecol - 1], dp

def Levenshtein_Distance_with_Transposition_Date_Rareness(seq1, seq2, dates1, dates2, dict_pre_matrix, max_transposition_date):
    """
    Calculates the Levenshtein distance between two sequences, considering transpositions.
    It is possible that distance > max_distance, if same events happen within trans date. (Vtns will be added.) 
    Args:
    trans_date = 0 # means turn off transpose
    seq1: The first sequence.
    seq2: The second sequence.
    dict_pre_matrix: dict that from rareness weighted rank
    Returns:
    The Levenshtein distance between the two sequences.
    """
    seq1 = ["#"] +  [x for x in seq1]
    seq2 = ["#"] +  [x for x in seq2]
    dates1 = ["None"] + [x for x in dates1]
    dates2 = ["None"] + [x for x in dates2]
    
    sizerow = len(seq1)
    sizecol = len(seq2)
    # Create a distance matrix
    dp = [[0 for col in range(sizecol)] for row in range(sizerow)]
    w1, w2, w3 = -0.5, 0.75, 0.5
    #in the manuscript, 𝑤1 = -0.5, 𝑤2 = 0.75, and 𝑤3 = 0.5
    ## Vsub = Vmtc * w1, (w1<=0)
    ## Vins = Vsub * w2 = Vmtc * w1*w2  ([0.5, 1])
    ## Vtns = Vmtc * w3  ([0,1])
    for i in range(1, sizerow):
        for j in range(1, sizecol):
            # Standard costs
            Vmtc_row = dict_pre_matrix[seq1[i]]["Vmtc"]
            Vmtc_col = dict_pre_matrix[seq2[j]]["Vmtc"]
            Vins_row = w1*w2*Vmtc_row
            Vins_col = w1*w2*Vmtc_col
            insertion_cost = dp[i][j-1] + min(Vins_row, Vins_col) # ( # -> letter)
            deletion_cost = dp[i-1][j] + min(Vins_row, Vins_col)  # (letter -> # )
            # Substitution cost
            min_dp = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]) #dp[i-1][j-1] #
            if (seq1[i]!= seq2[j]):
                sub_cost = min_dp + min(w1*Vmtc_row, w1*Vmtc_col)  # sub
            ## change sub_cost if same sequence
            if (seq1[i] == seq2[j]): ## sub or tns or mtc
                if (abs((dates1[i] - dates2[j]).days) >= max_transposition_date):
                    ## sub: same sequence, but out of date.
                    sub_cost = min_dp + min(w1*Vmtc_row, w1*Vmtc_col)
                else: ## within tns date,
                    sub_cost = min_dp + min(Vmtc_row, Vmtc_col)
                    if(i!=j):## tns
                        sub_cost = min_dp + min(w3*Vmtc_row, w3*Vmtc_col)
            dp[i][j] = round(min(insertion_cost, deletion_cost, sub_cost), 1)
    return dp[sizerow - 1][sizecol - 1], dp

def Normalize_Levenshtein_Distance_Score(seq1, seq2, dates1, dates2, dict_sub_matrix, max_transposition_date):
    ''' 0~1'''
    distance, matrix = Levenshtein_Distance_with_Transposition_Date_Beta(seq1, seq2, dates1, dates2, dict_sub_matrix, max_transposition_date)
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

def Normalize_Levenshtein_Distance_Score_Rareness(seq1, seq2, dates1, dates2, dict_rare_matrix, max_transposition_date):
    ''' 0~1'''
    distance, matrix = Levenshtein_Distance_with_Transposition_Date_Rareness(seq1, seq2, dates1, dates2, dict_rare_matrix, max_transposition_date)
    #print ("Leven_Distance", distance)

    seq1_penal = [dict_rare_matrix[char]['Vins'] for char in seq1] #[dict_matrix[char]["#"] for char in seq1]
    seq2_penal = [dict_rare_matrix[char]['Vins'] for char in seq2]# [dict_matrix[char]["#"] for char in seq2]
    
    seq1_match = [dict_rare_matrix[char]['Vmtc'] for char in seq1] #[dict_matrix[char][char] for char in seq1]
    seq2_match = [dict_rare_matrix[char]['Vmtc'] for char in seq2]

    distance_max = max(sum(seq1_penal), sum(seq2_penal))
    distance_min = min(sum(seq1_match[::-1]), sum(seq2_match[::-1]))
    normalized_score = (distance - distance_min)/ (distance_max-distance_min)
    if (distance_max == distance_min): normalized_score = 1
    similarity_score = 1 - normalized_score
    return similarity_score