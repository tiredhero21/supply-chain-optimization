def optimize_order(data, model, X_cols):
    data['Predicted_Demand'] = model.predict(data[X_cols])
    data['Order_Quantity'] = data['Predicted_Demand'] - data['Stock levels']
    data['Order_Quantity'] = data['Order_Quantity'].apply(lambda x: max(x, 0))
    return data
