import numpy as np



class TreeSplanerClassifier:
    """
    A class to extract and convert decision rules from a trained decision tree into natural language.
    """

    def __init__(self, clf, feature_names=None, target_names=None):
        self.clf = clf
        # Explicitly check if feature_names or target_names are None
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(clf.tree_.n_features)]
        else:
            self.feature_names = feature_names

        if target_names is None:
            self.target_names = [f"class_{i}" for i in range(clf.tree_.n_classes)]
        else:
            self.target_names = target_names

        # Cache commonly used tree attributes
        self.children_left = clf.tree_.children_left
        self.children_right = clf.tree_.children_right
        self.feature = clf.tree_.feature
        self.threshold = clf.tree_.threshold
        self.value = clf.tree_.value

    def find_paths_from_root(self, node=0, path=None, all_paths=None):
        """Recursively finds all paths from the root node to each leaf node."""
        if path is None:
            path = []
        if all_paths is None:
            all_paths = []
        
        path.append(node)

        # Check if it is a leaf node
        if self.children_left[node] == -1 and self.children_right[node] == -1:
            all_paths.append(list(path))
        else:
            # Recursively explore the left and right children
            if self.children_left[node] != -1:
                self.find_paths_from_root(self.children_left[node], path, all_paths)
            if self.children_right[node] != -1:
                self.find_paths_from_root(self.children_right[node], path, all_paths)
        
        path.pop()
        
        return all_paths

    def decision_tree_to_text(self):
        """
        Converts the tree's structure into a natural language description of its decision rules.
        """
        all_paths = self.find_paths_from_root()
        branch = "If "
        number_paths = len(all_paths)
   

        for p, path in enumerate(all_paths):
            branch += " ("
            for l, node in enumerate(path):
               

                
                # Determine inequality direction based on child position
                if node in self.children_right:
                    eq = ">"
                    
                
                elif node==0 and  path[l+1] in  self.children_right:
                    eq = ">"
                    
                else:
                    eq = "<="

                if l != len(path) - 1:
                    branch += f"{self.feature_names[self.feature[node]]} {eq} {round(self.threshold[node], 2)} and "
                else:
                    target_index=np.argmax(self.value[node, 0, :])

                    predicted_class = self.target_names[target_index]
                    probability_class=round(self.clf.tree_.value[node,0,target_index],3)
                    branch += f"{self.feature_names[self.feature[node]]} {eq} {round(self.threshold[node], 2)} then class is {predicted_class}  with probability of {probability_class}"
            branch += " )" + (" or" if p != number_paths - 1 else "")

        return branch
    

    def build_text_prediction(self, samples, target_index=None):
        """
        Generates natural language predictions for specific samples.
        """
        branches = []
    
        for sample in samples:
            sample = np.array(sample).reshape(1, -1)
            node_indicator = self.clf.decision_path(sample)
            predicted_index = int(self.clf.predict(sample)[0])
            predicted_class = self.target_names[predicted_index]
    
            # Set target based on target_index if provided
            if target_index is not None:
                target = self.target_names[int(target_index)]

    
            probability_class = np.round(self.clf.predict_proba(sample)[0][predicted_index], 3)
            branch = ""
    
            # Get the path for the current sample
            path = node_indicator.indices
            for i, node in enumerate(path):
                # Determine inequality direction based on child position
                if node in self.children_right:
                    eq = ">"
                else:
                    eq = "<="
    
                if i != len(path) - 1:
                    branch += f"{self.feature_names[self.feature[node]]} {eq} {round(self.threshold[node], 2)} and "
                else:
                    if target_index is None:
                        branch += f"{self.feature_names[self.feature[node]]} {eq} {round(self.threshold[node], 2)} therefore the predicted class is {predicted_class} with probability of {probability_class}"
                    else:
                        branch += f"{self.feature_names[self.feature[node]]} {eq} {round(self.threshold[node], 2)} therefore the predicted class is {predicted_class} with probability of {probability_class}, the true class is {target}"
    
            branches.append(branch)
    
        return branches

    def  branch_impurity(self):

        all_paths=self.find_paths_from_root()

        node_impurity=self.clf.tree_.impurity

        feature=self.feature
        
        branch_impurity=""
        for p, path in enumerate(all_paths):
            branch_impurity+=f" For branch {p+1} "
            
            for node in path:
        
                branch_impurity+=f" for split  {node } and feature {self.feature_names[feature[node]]} impurity is {round(node_impurity[node],3)} "
        
        return branch_impurity
