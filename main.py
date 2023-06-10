import tkinter as tk
from tkinter import messagebox, ttk

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sqlite3


class CancerSurvivalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cancer Survival Prediction")
        self.geometry("500x300")

        self.data = pd.read_csv('res/haberman.data', names=['Age', 'Year', 'Nodes', 'Status'], header=None)
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]

        self.knn = None
        self.model_file = "model.joblib"
        self.conn = sqlite3.connect('data.db')
        self.c = self.conn.cursor()
        self.create_widgets()

        self.load_model()

    def create_widgets(self):
        age_label = tk.Label(self, text="Age:")
        age_label.grid(row=0, column=0)
        self.age_entry = tk.Entry(self)
        self.age_entry.grid(row=0, column=1)

        year_label = tk.Label(self, text="Year of operation:")
        year_label.grid(row=1, column=0)
        self.year_entry = tk.Entry(self)
        self.year_entry.grid(row=1, column=1)

        nodes_label = tk.Label(self, text="Number of positive axillary nodes:")
        nodes_label.grid(row=2, column=0)
        self.nodes_entry = tk.Entry(self)
        self.nodes_entry.grid(row=2, column=1)

        status_label = tk.Label(self, text="Survival status (1 or 2):")
        status_label.grid(row=3, column=0)
        self.status_entry = tk.Entry(self)
        self.status_entry.grid(row=3, column=1)

        train_button = tk.Button(self, text="Train Model", command=self.train_model)
        train_button.grid(row=4, column=0)

        predict_button = tk.Button(self, text="Predict", command=self.predict)
        predict_button.grid(row=2, column=3)

        add_button = tk.Button(self, text="Add Data", command=self.add_data)
        add_button.grid(row=4, column=1)

        save_data_button = tk.Button(self, text="Save Data", command=self.save_data)
        save_data_button.grid(row=5, column=1)

        rebuild_button = tk.Button(self, text="Rebuild Model", command=self.rebuild_model)
        rebuild_button.grid(row=5, column=0)

        visualize_button = tk.Button(self, text="Visualize Data", command=self.visualize_data)
        visualize_button.grid(row=6, column=1)

        save_model_button = tk.Button(self, text="Save Model", command=self.save_model)
        save_model_button.grid(row=6, column=0)

        browse_button = tk.Button(self, text="Browse Data", command=self.browse_data)
        browse_button.grid(row=7, column=1)

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.knn = KNeighborsClassifier()
        self.knn.fit(X_train, y_train)
        y_pred = self.knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        messagebox.showinfo("Model Evaluation", f"Accuracy: {accuracy}")

    def predict(self):
        age = int(self.age_entry.get())
        year = int(self.year_entry.get())
        nodes = int(self.nodes_entry.get())
        new_data = [[age, year, nodes]]
        new_data_df = pd.DataFrame(new_data, columns=['Age', 'Year', 'Nodes'])
        prediction = self.knn.predict(new_data_df)[0]
        if prediction == 1:
            result = "Survived 5 years or longer"
        else:
            result = "Died within 5 years"
        messagebox.showinfo("Prediction Result", result)

    def add_data(self):
        age = int(self.age_entry.get())
        year = int(self.year_entry.get())
        nodes = int(self.nodes_entry.get())
        status = int(self.status_entry.get())
        new_data = pd.DataFrame([[age, year, nodes, status]], columns=['Age', 'Year', 'Nodes', 'Status'])
        self.data = pd.concat([self.data, new_data], ignore_index=True)
        self.c.execute("INSERT INTO patients VALUES (?, ?, ?, ?)", (age, year, nodes, status))
        self.conn.commit()
        messagebox.showinfo("Success", "Data added to the program and the database")

    def save_data(self):
        self.data.to_csv('res/haberman.data', index=False, header=False)
        messagebox.showinfo("Success", "Data saved to 'haberman.data'")

    def rebuild_model(self):
        self.c.execute("SELECT * FROM patients")
        rows = self.c.fetchall()
        data = pd.DataFrame(rows, columns=['age', 'year', 'nodes', 'status'])
        self.X = data.iloc[:, :-1]
        self.y = data.iloc[:, -1]
        self.knn = KNeighborsClassifier()
        self.knn.fit(self.X, self.y)
        messagebox.showinfo("Success", "Model rebuilt using new data")

    def save_model(self):
        if self.knn is not None:
            joblib.dump(self.knn, self.model_file)
            messagebox.showinfo("Success", "Model saved to 'model.joblib'")
        else:
            messagebox.showwarning("Warning", "No model to save")

    def load_model(self):
        try:
            self.knn = joblib.load(self.model_file)
        except FileNotFoundError:
            messagebox.showwarning("Warning", "No model found")

    def visualize_data(self):
        plt.scatter(self.data['Age'], self.data['Nodes'], c=self.data['Status'])
        plt.xlabel('Age')
        plt.ylabel('Number of positive axillary nodes')
        plt.show()

    def browse_data(self):
        browse_window = tk.Toplevel(self)
        browse_window.title("Browse Data")
        tree = ttk.Treeview(browse_window)
        tree["columns"] = ("Age", "Year", "Nodes", "Status")
        tree.column("#0", width=0, stretch=tk.NO)
        tree.column("Age", anchor=tk.CENTER, width=100)
        tree.column("Year", anchor=tk.CENTER, width=100)
        tree.column("Nodes", anchor=tk.CENTER, width=150)
        tree.column("Status", anchor=tk.CENTER, width=150)
        tree.heading("#0", text="", anchor=tk.CENTER)
        tree.heading("Age", text="Age", anchor=tk.CENTER)
        tree.heading("Year", text="Year", anchor=tk.CENTER)
        tree.heading("Nodes", text="Nodes", anchor=tk.CENTER)
        tree.heading("Status", text="Status", anchor=tk.CENTER)
        for index, row in self.data.iterrows():
            tree.insert("", tk.END, text=index, values=(row['Age'], row['Year'], row['Nodes'], row['Status']))
        scrollbar = ttk.Scrollbar(browse_window, orient="vertical", command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)


if __name__ == "__main__":
    app = CancerSurvivalApp()
    app.mainloop()
