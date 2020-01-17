from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics

# Confusion matrix, sensitivity and specificity
y_prob = model.predict(X_test) 
y_classes = y_prob
y_classes[y_classes<0.5]=0
y_classes[y_classes!=0]=1

cm1 = confusion_matrix(y_test[:,1], y_classes)
print("Confusion matrix:")
print(cm1)

specificity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Specificity: ', specificity1)

sensitivity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Sensitivity: ', sensitivity1 )

precision1 = cm1[1,1]/(cm1[0,1]+cm1[1,1])
print('Precision: ', precision1)

acc = np.mean(y_classes==y_test[:,1])
print('Accuracy: ', acc)

# ROC curve and AUC
scores = y_prob
fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
