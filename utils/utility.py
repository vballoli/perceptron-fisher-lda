import matplotlib.pyplot as plt
import seaborn as sns


def log(should_print=False, separator='\t', *args):
    if should_print:
        print(*args)
        
def visualize_lda():
    pass

def normal_plot():
    pass

def scatterplot(data, X_label, y_label, class_labels):
    sns.set()
    sns.scatterplot(x=X_label, y=y_label, hue="day", style="time", data=data)