import joblib
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse

modelo = joblib.load('modelos/calibrated_xgb_model.joblib')
scaler = joblib.load('modelos/feature_scaler.joblib')
features = joblib.load('modelos/feature_metadata.joblib')

def evaluar_riesgo(request):
    if request.method == 'POST':
        datos = {
            'RevolvingUtilizationOfUnsecuredLines': float(request.POST.get('RevolvingUtilizationOfUnsecuredLines')),
            'age': int(request.POST.get('age')),
            'NumberOfTime30-59DaysPastDueNotWorse': int(request.POST.get('NumberOfTime30-59DaysPastDueNotWorse')),
            'DebtRatio': float(request.POST.get('DebtRatio')),
            'MonthlyIncome': float(request.POST.get('MonthlyIncome')),
            'NumberOfOpenCreditLinesAndLoans': int(request.POST.get('NumberOfOpenCreditLinesAndLoans')),
            'NumberOfTimes90DaysLate': int(request.POST.get('NumberOfTimes90DaysLate')),
            'NumberRealEstateLoansOrLines': int(request.POST.get('NumberRealEstateLoansOrLines')),
            'NumberOfTime60-89DaysPastDueNotWorse': int(request.POST.get('NumberOfTime60-89DaysPastDueNotWorse')),
            'NumberOfDependents': int(request.POST.get('NumberOfDependents'))
        }

        df = pd.DataFrame([datos])
        
        df['TotalLateEvents'] = df['NumberOfTime30-59DaysPastDueNotWorse'] + df['NumberOfTime60-89DaysPastDueNotWorse'] + df['NumberOfTimes90DaysLate']
        df['MonthlyDebt'] = df['DebtRatio'] * df['MonthlyIncome']
        df['DebtToIncomePerDependent'] = df['MonthlyDebt'] / (df['NumberOfDependents'] + 1)
        df['HighRevolvingUtilization'] = int(df['RevolvingUtilizationOfUnsecuredLines'].iloc[0] > 1)
        
        if isinstance(features, dict):
            columnas_esperadas = features.get('feature_names', [])
        elif isinstance(features, list):
            columnas_esperadas = features
        else:
            columnas_esperadas = features.columns
            
        df = df.reindex(columns=columnas_esperadas, fill_value=0)

        datos_escalados = scaler.transform(df)
        probabilidad = modelo.predict_proba(datos_escalados)[0][1]
        
        es_riesgoso = probabilidad >= 0.10
        resultado = "CRÉDITO RECHAZADO" if es_riesgoso else "CRÉDITO APROBADO"

        return JsonResponse({
            'resultado': resultado,
            'probabilidad': round(probabilidad * 100, 2),
            'clase': 'danger' if es_riesgoso else 'success'
        })

    return render(request, 'formulario.html')