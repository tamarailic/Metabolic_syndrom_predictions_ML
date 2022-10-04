import tkinter as tk
import numpy as np
import pickle as pc

def get_inputs():
    x = []
    for entry in entries:
        x.append(entry.get())

    for value in radio_values:
        x.append(value.get())

    case = np.array([[float(x[0]),x[8],x[9],x[10],x[11],float(x[6]),x[7],float(x[1]),float(x[2]),int(x[3]),int(x[4]),float(x[5]),x[12]]])
    case_sci = np.array([[float(x[6]),float(x[1]), int(x[3]),int(x[4]),float(x[2]),x[12]]])
    case_stat = np.array([[float(x[0]),x[9], float(x[6]),float(x[1]),int(x[3]),float(x[2]), x[12]]])
    predict_results(case,case_sci,case_stat)

def predict_results(X,X_sci,X_stat):
    results = []
    with open("data/lr_best.pickle","rb") as file:
        lr_best = pc.load(file)
    with open("data/lr_best_sci.pickle","rb") as file:
        lr_best_sci = pc.load(file)
    with open("data/lr_best_stat.pickle","rb") as file:
        lr_best_stat = pc.load(file)

    results.append(lr_best.predict(X))
    results.append(lr_best_sci.predict(X_sci))
    results.append(lr_best_stat.predict(X_stat))

    with open("data/tree_best.pickle","rb") as file:
        tree_best = pc.load(file)
    with open("data/tree_best_sci.pickle","rb") as file:
        tree_best_sci = pc.load(file)
    with open("data/tree_best_stat.pickle","rb") as file:
        tree_best_stat = pc.load(file)

    results.append(tree_best.predict(X))
    results.append(tree_best_sci.predict(X_sci))
    results.append(tree_best_stat.predict(X_stat))

    with open("data/forest_best.pickle","rb") as file:
        forest_best = pc.load(file)
    with open("data/forest_best_sci.pickle","rb") as file:
        forest_best_sci = pc.load(file)
    with open("data/forest_best_stat.pickle","rb") as file:
        forest_best_stat = pc.load(file)

    results.append(forest_best.predict(X))
    results.append(forest_best_sci.predict(X_sci))
    results.append(forest_best_stat.predict(X_stat))

    with open("data/knns_best.pickle","rb") as file:
        knns_best = pc.load(file)
    with open("data/knns_best_sci.pickle","rb") as file:
        knns_best_sci = pc.load(file)
    with open("data/knns_best_stat.pickle","rb") as file:
        knns_best_stat = pc.load(file)

    results.append(knns_best.predict(X))
    results.append(knns_best_sci.predict(X_sci))
    results.append(knns_best_stat.predict(X_stat))

    with open("data/svm_best.pickle","rb") as file:
        best_model = pc.load(file)
    with open("data/svm_best_sci.pickle","rb") as file:
        best_model_sci = pc.load(file)
    with open("data/svm_best_stat.pickle","rb") as file:
        best_model_stat = pc.load(file)

    results.append(best_model.predict(X))
    results.append(best_model_sci.predict(X_sci))
    results.append(best_model_stat.predict(X_stat))

    results = ["True" if i[0] == 1 else "False" for i in results]
    result_window(results)

def result_window(results):
    result_win = tk.Toplevel(base)
    result_win.title("Rezultati")
    result_win.geometry("400x500")
    result_frame = tk.Frame(result_win)
    result_frame.pack(padx=5, pady=5)
    labels = ["Loigistic Regression: ","Loigistic Regression Science: ","Loigistic Regression Statistic: ",
              "Decision tree: ","Decision tree Science: ","Decision tree Statistic: ",
              "RandomForest: ","RandomForest Science: ","RandomForest Statistic: ",
              "KNN: ","KNN Science: ","KNN Statistic: ",
              "SVM: ","SVM Science: ","SVM Statistic: ",]
    for i in range(15):
        model = tk.Frame(result_frame)
        model.pack()
        l = tk.Label(model,text=labels[i]+results[i])
        l.pack(side = tk.LEFT,padx=5, pady=5)

if __name__ == '__main__':

    base = tk.Tk()
    base.title('Predviđanje metaboličkog sindroma')
    base.geometry("450x650")
    window = tk.Frame(master=base)
    window.pack(fill=tk.BOTH, expand=True,padx=20, pady=20)
    heading_frame = tk.Label(master=window,text='Predviđanje metaboličkog sindroma')
    heading_frame.pack(expand=True, fill=tk.BOTH)
    input_frame = tk.Frame(master=window,padx=20, pady=20)
    input_frame.pack(fill=tk.BOTH, expand=True)
    label_frame = tk.Frame(master = input_frame)
    label_frame.pack(side = tk.LEFT)
    labels = ['UZRAST', 'TRIGLICERIDI', 'HDL', 'SISTOLNI PRITISAK', 'DIJASTOLNI PRITISAK', 'ŠUK', 'OS(cm)', 'OS (percentil)', 'GOJAZNOST (R)', 'HIPERTENZIJA (R)', 'ŠEĆERNA BOLEST (R)', 'HIPERLIPIDEMIJA (R)', 'ŠEĆERNA BOLEST']
    entry_frame = tk.Frame(master = input_frame,pady=20)
    entry_frame.pack(side = tk.RIGHT)
    entries = []
    for label in labels[:7]:
        l = tk.Label(master=label_frame, text=label, justify=tk.LEFT)
        e = tk.Entry(master=entry_frame, width=15)
        entries.append(e)
        e.pack(padx=5, pady=7)
        l.pack(padx=5, pady=5)

    radio_values = []
    buttons_frame = tk.Frame(master=entry_frame)
    buttons_frame.pack()
    var = tk.IntVar()
    var.set(0)
    radio_values.append(var)
    l = tk.Label(master=label_frame, text='OS (percentil)')
    R1 = tk.Radiobutton(buttons_frame, text="50", variable=var, value=0)
    R1.pack(side="left", padx=5, pady=5)
    R2 = tk.Radiobutton(buttons_frame, text="75", variable=var, value=1)
    R2.pack(side="left", padx=5, pady=5)
    R3 = tk.Radiobutton(buttons_frame, text=">90", variable=var, value=2)
    R3.pack(side="left",padx=5, pady=5)
    l.pack(padx=5, pady=5)

    for label in labels[8:]:
        buttons_frame = tk.Frame(master=entry_frame)
        buttons_frame.pack()
        var = tk.IntVar()
        var.set(0)
        radio_values.append(var)
        l = tk.Label(master=label_frame, text=label)
        R1 = tk.Radiobutton(buttons_frame, text="0", variable=var, value=0)
        R1.pack(side="left",padx=5, pady=5)
        R2 = tk.Radiobutton(buttons_frame, text="1", variable=var, value=1)
        R2.pack(side="left",padx=5, pady=5)
        l.pack(padx=5, pady=5)

    submit_frame = tk.Frame(master=window)
    submit_frame.pack(fill=tk.BOTH, expand=True,padx=5, pady=5)
    submit = tk.Button(master=submit_frame, text="Predvidi", command=get_inputs)
    submit.pack(padx=5, pady=5)

    window.mainloop()