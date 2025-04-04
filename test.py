# ROC-кривые
y_pred_prob_rfc = rfc.predict_proba(X_test)[:, 1]
y_pred_prob_nn = mlp.predict_proba(X_test)[:, 1] 


fpr_rfc, tpr_rfc, _ = roc_curve(y_test, y_pred_prob_rfc)
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_prob_nn)
roc_auc_rfc = auc(fpr_rfc, tpr_rfc)
roc_auc_nn = auc(fpr_nn, tpr_nn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rfc, tpr_rfc, label=f'Random Forest Tree (AUC = {roc_auc_rfc:.2f})', color='blue')
plt.plot(fpr_nn, tpr_nn, label=f'Neural Network (AUC = {roc_auc_nn:.2f})', color='red')
plt.plot([0, 1], [0, 1], 'k--')  # Линия случайной модели
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()