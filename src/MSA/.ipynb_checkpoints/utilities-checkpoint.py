import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class Person:
    """Represents a person with attributes like name, age, etc."""
    def __init__(self, name, age=-1, gender="", story="", event=""):
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
        self.events = event

    def set_story(self, df_events, col_events, col_dates):
         """
         Take input from df one column for events, another for date.
         """
         self.events = ''.join(df_events.loc[:, col_events])
         self.dates = df_events.loc[:, col_dates].values
     
    def introduce(self):
        """Returns a string introducing the person."""
        intro = f"I'm {self.name}, {self.age} years old, {self.gender}. Events: "
        if (self.story):
           intro += f"\n {self.story}"
        elif(self.events):
           intro += f"\n {self.events}"
        return intro

    def visualize_history(self):
        releases = self.events
        dates = self.dates
        
        i = 0
        levels = [1.0]
        date_previous = dates[0]
        for date in dates[1:]:
            date_diff = abs((date - date_previous).days)
            if (date_diff <=0):
                i += 1  ## same date so increase levels
            else:
                i = 0
            levels.append(1+i*0.1)
            date_previous = date
        
        # The figure and the axes.
        fig, ax = plt.subplots(figsize=(8.8, 4), layout="constrained")
        ax.set(title=self.name + "_Timeline")
        
        # The vertical stems.
        ax.vlines(self.dates, 0, levels,
                  color=[("tab:red", 1 if release.endswith(".0") else .5)
                         for release in releases])
        # The baseline.
        ax.axhline(0, c="green")
        
        # The markers on the baseline.
        minor_dates = [date for date, release in zip(dates, releases) if release[-1] == '0']
        bugfix_dates = [date for date, release in zip(dates, releases) if release[-1] != '0']
        ax.plot(bugfix_dates, np.zeros_like(bugfix_dates), "ko", mfc="white")
        ax.plot(minor_dates, np.zeros_like(minor_dates), "ko", mfc="tab:red")
        
        # Annotate the lines.
        for date, level, release in zip(dates, levels, releases):
            ax.annotate(release, xy=(date, level),
                        xytext=(-3, np.sign(level)), textcoords="offset points",
                        verticalalignment="bottom" if level > 0 else "top",
                        weight="bold" if release.endswith(".0") else "normal",
                        bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))
        
        ax.yaxis.set(major_locator=mdates.YearLocator(),
                     major_formatter=mdates.DateFormatter("%Y"))
        
        # Remove the y-axis and some spines.
        ax.yaxis.set_visible(False)
        ax.spines[["left", "top", "right"]].set_visible(False)
        
        ax.margins(y=0.1)
        plt.show()

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
    
class ClassName:
    """Docstring explaining the class purpose"""
    
    def __init__(self, arg1, arg2):
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
    
    def method1(self, arg1, arg2):
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
