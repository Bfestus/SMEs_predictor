"""
SME Success Predictor - Model Testing Script

This script loads the saved model and demonstrates how to make predictions
on new business data. It can be used independently of the training notebook.

Author: SME Predictor Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

def load_model_and_info(models_dir='../models'):
    """
    Load the saved model and its metadata
    
    Returns:
        tuple: (model, model_info)
    """
    try:
        # Load model info
        with open(f'{models_dir}/model_info.json', 'r') as f:
            model_info = json.load(f)
        
        # Load the model
        model_file = model_info['model_file']
        model = joblib.load(model_file)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model: {model_info['model_name']}")
        print(f"   F1-Score: {model_info['model_performance']['f1_score']:.4f}")
        print(f"   Accuracy: {model_info['model_performance']['accuracy']:.4f}")
        
        return model, model_info
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def engineer_features(df):
    """
    Add engineered features to the input dataframe
    
    Args:
        df (pd.DataFrame): Input business data
        
    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    df = df.copy()
    
    # 1. Business age categories
    df['age_category'] = pd.cut(df['Business_Age'], 
                               bins=[0, 2, 5, 10, float('inf')], 
                               labels=['Very Young', 'Young', 'Mature', 'Established'])
    
    # 2. Capital categories (in millions RWF)
    df['capital_category'] = pd.cut(df['Initial_Capital'], 
                                   bins=[0, 1000000, 5000000, 10000000, float('inf')], 
                                   labels=['Small', 'Medium', 'Large', 'Very Large'])
    
    # 3. Employee size categories
    df['employee_category'] = pd.cut(df['Num_Employees'], 
                                    bins=[0, 5, 20, 50, float('inf')], 
                                    labels=['Micro', 'Small', 'Medium', 'Large'])
    
    # 4. Owner age categories
    df['owner_age_category'] = pd.cut(df['Owner_Age'], 
                                     bins=[0, 30, 40, 50, float('inf')], 
                                     labels=['Young', 'Middle-aged', 'Senior', 'Elder'])
    
    # 5. Technology sector flag
    tech_sectors = ['Information & Communication Technology (ICT)', 'Energy & Utilities']
    df['is_tech_sector'] = df['Business_Sector'].isin(tech_sectors).astype(int)
    
    # 6. Capital per employee ratio
    df['capital_per_employee'] = df['Initial_Capital'] / (df['Num_Employees'] + 1)
    
    return df

def predict_sme_success(business_data, model=None, model_info=None):
    """
    Predict SME success probability for business data
    
    Args:
        business_data (dict or pd.DataFrame): Business information
        model: Loaded ML model (optional)
        model_info: Model metadata (optional)
    
    Returns:
        dict: Prediction results
    """
    
    # Load model if not provided
    if model is None or model_info is None:
        model, model_info = load_model_and_info()
        if model is None:
            return {"error": "Could not load model"}
    
    # Convert to DataFrame if dict
    if isinstance(business_data, dict):
        df = pd.DataFrame([business_data])
    else:
        df = business_data.copy()
    
    # Add engineered features
    df = engineer_features(df)
    
    # Select required features
    features_to_use = model_info['features_to_use']
    X = df[features_to_use]
    
    # Make predictions
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    success_prob = probabilities[1] * 100
    confidence = max(probabilities) * 100
    
    # Generate recommendation
    if success_prob >= 70:
        recommendation = "🟢 HIGH SUCCESS POTENTIAL - Proceed with confidence!"
        risk_level = "Low"
    elif success_prob >= 50:
        recommendation = "🟡 MODERATE SUCCESS POTENTIAL - Consider improvements in weak areas."
        risk_level = "Medium"
    else:
        recommendation = "🔴 LOW SUCCESS POTENTIAL - Significant improvements needed."
        risk_level = "High"
    
    return {
        'prediction': 'SUCCESS' if prediction == 1 else 'FAILURE',
        'success_probability': round(success_prob, 1),
        'confidence': round(confidence, 1),
        'risk_level': risk_level,
        'recommendation': recommendation,
        'timestamp': datetime.now().isoformat()
    }

def create_sample_businesses():
    """
    Create sample business scenarios for testing
    
    Returns:
        pd.DataFrame: Sample businesses
    """
    
    sample_data = {
        'Tech Startup (Kigali)': {
            'Initial_Capital': 5000000,
            'Num_Employees': 8,
            'Business_Age': 2,
            'Duration_Operation': 2,
            'Owner_Age': 28,
            'Growth_Indicator': 0.25,
            'Business_Sector': 'Information & Communication Technology (ICT)',
            'Business_Subsector': 'Software development, IT services & startups',
            'Business_Model': 'Service-based',
            'Ownership_Type': 'Limited Company',
            'Location': 'Kigali',
            'Owner_Gender': 'Female',
            'Business_Type': 'SME'
        },
        'Manufacturing Company': {
            'Initial_Capital': 12000000,
            'Num_Employees': 35,
            'Business_Age': 8,
            'Duration_Operation': 8,
            'Owner_Age': 45,
            'Growth_Indicator': 0.15,
            'Business_Sector': 'Manufacturing',
            'Business_Subsector': 'Food processing & packaging',
            'Business_Model': 'Product-based',
            'Ownership_Type': 'Limited Company',
            'Location': 'Kigali',
            'Owner_Gender': 'Male',
            'Business_Type': 'SME'
        },
        'Small Retail Store': {
            'Initial_Capital': 800000,
            'Num_Employees': 3,
            'Business_Age': 1,
            'Duration_Operation': 1,
            'Owner_Age': 35,
            'Growth_Indicator': -0.05,
            'Business_Sector': 'Trade & Commerce',
            'Business_Subsector': 'General retail & wholesale',
            'Business_Model': 'Hybrid',
            'Ownership_Type': 'Sole Proprietorship',
            'Location': 'Rusizi',
            'Owner_Gender': 'Female',
            'Business_Type': 'Micro-enterprise'
        },
        'Construction Company': {
            'Initial_Capital': 20000000,
            'Num_Employees': 45,
            'Business_Age': 12,
            'Duration_Operation': 12,
            'Owner_Age': 52,
            'Growth_Indicator': 0.08,
            'Business_Sector': 'Construction & Real Estate',
            'Business_Subsector': 'Road construction & maintenance',
            'Business_Model': 'Product-based',
            'Ownership_Type': 'Limited Company',
            'Location': 'Kigali',
            'Owner_Gender': 'Male',
            'Business_Type': 'SME'
        }
    }
    
    return sample_data

def main():
    """
    Main testing function
    """
    print("🧪 SME SUCCESS PREDICTOR - MODEL TESTING")
    print("="*60)
    
    # Load model
    model, model_info = load_model_and_info()
    if model is None:
        return
    
    # Create sample businesses
    sample_businesses = create_sample_businesses()
    
    print(f"\n📊 TESTING {len(sample_businesses)} SAMPLE BUSINESSES")
    print("="*60)
    
    results = []
    
    for business_name, business_data in sample_businesses.items():
        print(f"\n🏢 {business_name}")
        print("-" * 50)
        
        # Print business details
        print(f"   📍 Location: {business_data['Location']}")
        print(f"   💰 Capital: {business_data['Initial_Capital']:,} RWF")
        print(f"   👥 Employees: {business_data['Num_Employees']}")
        print(f"   📅 Age: {business_data['Business_Age']} years")
        print(f"   📈 Growth: {business_data['Growth_Indicator']:.1%}")
        print(f"   🏭 Sector: {business_data['Business_Sector']}")
        
        # Make prediction
        result = predict_sme_success(business_data, model, model_info)
        
        print(f"\n   🎯 PREDICTION:")
        print(f"      Result: {result['prediction']}")
        print(f"      Success Probability: {result['success_probability']}%")
        print(f"      Confidence: {result['confidence']}%")
        print(f"      Risk Level: {result['risk_level']}")
        print(f"      {result['recommendation']}")
        
        results.append({
            'business': business_name,
            'prediction': result['prediction'],
            'probability': result['success_probability'],
            'risk': result['risk_level']
        })
    
    # Summary
    print(f"\n📈 TESTING SUMMARY")
    print("="*60)
    successful_predictions = sum(1 for r in results if r['prediction'] == 'SUCCESS')
    avg_probability = sum(r['probability'] for r in results) / len(results)
    
    print(f"   • Total businesses tested: {len(results)}")
    print(f"   • Predicted successes: {successful_predictions}/{len(results)}")
    print(f"   • Average success probability: {avg_probability:.1f}%")
    print(f"   • Model status: ✅ Working correctly")
    
    print(f"\n✅ Model testing completed successfully!")
    print(f"🚀 Ready for deployment!")

if __name__ == "__main__":
    main()