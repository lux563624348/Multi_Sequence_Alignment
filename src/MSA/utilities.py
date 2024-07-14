import numpy as np
import pandas as pd

class SubstitutionMatrix:
    """
    This class represents a substitution matrix for a given number of events.
    """
    def __init__(self, num_events):
        """
        Initializes the substitution matrix with 0
        """
        self.num_events = num_events + 2
        if not isinstance(num_events, int) or num_events <= 0:
            raise ValueError("Number of events must be a positive integer.")
        self.matrix = np.full((self.num_events, self.num_events), 0)
      
    def set_index(self, indexs):
        self.indexs = ["#"]+ indexs + ["*"]  #
    
    def read_matrix_from_file(self, filename):
        """Reads the matrix from a text file, assuming one row per line."""
        try:
            with open(filename, 'r') as file:
                for line in file:
                    row = list(map(float, line.strip().split()))
                    self.matrix.append(row)
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            
    def iniate_scores(self, match_score, unmatch_score):
        self.matrix += np.diag(np.full(self.num_events, match_score - unmatch_score))
        self.matrix += np.full((self.num_events, self.num_events), unmatch_score)
        
    def set_score(self, event1, event2, score):
        self.df_matrix.loc[event1, event2] = score

    def set_score_2ways(self, event1, event2, score):
        self.df_matrix.loc[event1, event2] = score    
        self.df_matrix.loc[event2, event1] = score
        
    def read_matrix_from_df(self, df):
        self.df_matrix = df
    
    def to_df(self):
        self.df_matrix = pd.DataFrame(data = self.matrix, index = self.indexs, columns = self.indexs)
    
    def save_matrix_txt(self, path_out):
        np.savetxt(path_out, self.matrix, fmt = '%.1f')
    
    def save_df_matrix_txt(self, path_out):
        self.df_matrix.to_csv(path_out, sep='\t')
        
    def __str__(self):
        """
        Returns a string representation of the substitution matrix.
        """
        return self.df_matrix
    
class Person:
  """Represents a person with attributes like name, age, etc."""

  def __init__(self, name, age=18, gender="", story="", sequence=0):
    """
    Initializes a Person object with the given attributes.

    Args:
      name: The person's name (str)
      age: The person's age (int)
      gender: The person's gender (str)
      story: The person's story (str, optional)
      sequence: A sequence number (int, optional)
    """
    self.name = name
    self.age = age
    self.gender = gender
    self.story = story
    self.sequence = sequence

  def introduce(self):
    """Returns a string introducing the person."""
    intro = f"Hi, I'm {self.name}. I am {self.age} years old and identify as {self.gender}."
    if self.story:
      intro += f"\n {self.story}"
    return intro


class ClassName:
  """Docstring explaining the class purpose"""

  def __init__(self, arg1, arg2, ...):
    """
    Initialization method (constructor) for the class.

    Args:
      arg1: Description of the first argument
      arg2: Description of the second argument
      ...
    """
    self.attribute1 = arg1
    self.attribute2 = arg2
    # ... Initialize other attributes

  def method1(self, arg1, arg2, ...):
    """Docstring explaining the method's functionality

    Args:
      arg1: Description of the first argument
      arg2: Description of the second argument
      ...

    Returns:
      Description of the returned value (if applicable)
    """
    # Method body with functionality using the object's attributes
    return something  # Optional: return value
