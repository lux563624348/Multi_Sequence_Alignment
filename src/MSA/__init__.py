def Levenshtein_Distance_with_Transposition_Date_No_Matrix(seq1, seq2, dates1, dates2, dict_pre_matrix, max_transposition_date):
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
            insertion_cost = dp[i][j-1] + w1*w2*min(Vmtc_row, Vmtc_col)  # ( # -> letter)
            deletion_cost = dp[i-1][j] + w1*w2*min(Vmtc_row, Vmtc_col)  # (letter -> # )
            # Substitution cost
            min_dp =  min(dp[i][j-1], dp[i-1][j-1], dp[i-1][j]) #dp[i-1][j-1]
            if (seq1[i] != seq2[j]):
                sub_cost = min_dp + w1*min(Vmtc_row, Vmtc_col)  # sub
            ## change sub_cost if same sequence
            if (seq1[i] == seq2[j]): ## sub or tns or mtc
                if (abs((dates1[i] - dates2[j]).days) >= max_transposition_date):
                    ## sub: same sequence, but out of date.
                    sub_cost = min_dp + w1*min(Vmtc_row, Vmtc_col)
                else: ## within tns date,
                    if(i!=j):## tns
                        sub_cost = min_dp + w3*min(Vmtc_row, Vmtc_col)
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
            
            insertion_cost = dp[i][j-1] + min(dict_sub_matrix["#"][seq1[i]], dict_sub_matrix["#"][seq2[j]])  # ( # -> letter)
            deletion_cost = dp[i-1][j] + min(dict_sub_matrix[seq1[i]]["#"], dict_sub_matrix[seq2[j]]["#"])  # (letter -> # )
            # Substitution cost
            min_dp =  min(dp[i][j-1], dp[i-1][j-1], dp[i-1][j])
            sub_cost = min_dp + min(dict_sub_matrix[seq1[i]][seq2[j]], dict_sub_matrix[seq2[j]][seq1[i]])
            if (seq1[i] == seq2[j]): ## sub or tns or mtc
                Vmtc = dict_sub_matrix[seq1[i]][seq2[j]]
                if (abs((dates1[i] - dates2[j]).days) >= max_transposition_date):
                    ## sub: same sequence, but out of date.
                    sub_cost = min_dp + w1*Vmtc
                else: ## within tns date,
                    if(i!=j):## tns
                        sub_cost = min_dp + w3*Vmtc
            dp[i][j] = min(insertion_cost, deletion_cost, sub_cost)
    return dp[sizerow - 1][sizecol - 1], dp