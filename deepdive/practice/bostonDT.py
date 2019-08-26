#bostonDT.py

# from mglearn.plots import plot_animal_tree
# plot_animal_tree()
# plt.show()

from practice import bostonData, bostonReg
from sklearn.tree import export_graphviz
import graphviz

'''
export_graphviz(dt_regr, out_file='boston.dot',		#학습모델, 파일
                class_names=bostonData.label,	    #라벨, 타겟, 종속변수
	            feature_names=bostonData.columns,	#컬럼
	            impurity=False,			            #gini 미출력
	            filled=True)			            #node의 색깔을 다르게

with open('boston.dot') as file_reader:
    dot_graph = file_reader.read()
    dot = graphviz.Source(dot_graph)			#dot_graph의 source 저장
    dot.render(filename='boston.png')			#png로 저장
'''
