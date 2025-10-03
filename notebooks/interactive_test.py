"""
Interactive SME Success Predictor - Custom Input Testing

This script allows you to input your own business data and get predictions
from the trained SME success prediction model.

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
    """Load the saved model and its metadata"""
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
    """Add engineered features to the input dataframe"""
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

def predict_sme_success(business_data, model, model_info):
    """Predict SME success probability for business data"""
    
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
        advice = "Your business shows strong indicators for success. Focus on execution and scaling."
    elif success_prob >= 50:
        recommendation = "🟡 MODERATE SUCCESS POTENTIAL - Consider improvements in weak areas."
        risk_level = "Medium"
        advice = "Consider strengthening key areas like capital, location, or sector choice before proceeding."
    else:
        recommendation = "🔴 LOW SUCCESS POTENTIAL - Significant improvements needed."
        risk_level = "High"
        advice = "Review and significantly improve your business plan, especially capital, sector, and timing."
    
    return {
        'prediction': 'SUCCESS' if prediction == 1 else 'FAILURE',
        'success_probability': round(success_prob, 1),
        'confidence': round(confidence, 1),
        'risk_level': risk_level,
        'recommendation': recommendation,
        'advice': advice,
        'timestamp': datetime.now().isoformat()
    }

def get_user_input():
    """Get business data from user input"""
    print("\n" + "="*60)
    print("🏢 ENTER YOUR BUSINESS DETAILS")
    print("="*60)
    
    business_data = {}
    
    # Basic business information
    print("\n📋 Basic Business Information:")
    business_data['Initial_Capital'] = float(input("Initial Capital (RWF): "))
    business_data['Num_Employees'] = int(input("Number of Employees: "))
    business_data['Business_Age'] = int(input("Business Age (years): "))
    business_data['Duration_Operation'] = int(input("Duration of Operation (years): "))
    business_data['Growth_Indicator'] = float(input("Growth Indicator (-1.0 to 1.0, e.g., 0.15 for 15% growth): "))
    
    # Owner information
    print("\n👤 Owner Information:")
    business_data['Owner_Age'] = int(input("Owner Age: "))
    
    print("\nOwner Gender:")
    print("1. Male")
    print("2. Female")
    gender_choice = input("Choose (1 or 2): ")
    business_data['Owner_Gender'] = 'Male' if gender_choice == '1' else 'Female'
    
    # Business sector
    print("\n🏭 Business Sector:")
    sectors = [
        'Information & Communication Technology (ICT)',
        'Manufacturing',
        'Agriculture, Forestry & Fishing',
        'Construction & Real Estate',
        'Trade & Commerce',
        'Hospitality & Tourism',
        'Energy & Utilities',
        'Healthcare & Social Services',
        'Education & Training',
        'Financial Services',
        'Transport & Logistics',
        'Mining & Quarrying'
    ]
    
    for i, sector in enumerate(sectors, 1):
        print(f"{i:2d}. {sector}")
    
    sector_choice = int(input("Choose sector (1-12): ")) - 1
    business_data['Business_Sector'] = sectors[sector_choice]
    
    # Business subsector (simplified)
    business_data['Business_Subsector'] = input("Business Subsector (describe your specific area): ")
    
    # Business model
    print("\n💼 Business Model:")
    print("1. Product-based")
    print("2. Service-based")
    print("3. Hybrid")
    model_choice = input("Choose (1, 2, or 3): ")
    models = {'1': 'Product-based', '2': 'Service-based', '3': 'Hybrid'}
    business_data['Business_Model'] = models.get(model_choice, 'Hybrid')
    
    # Ownership type
    print("\n🏛️ Ownership Type:")
    print("1. Sole Proprietorship")
    print("2. Partnership")
    print("3. Limited Company")
    print("4. Cooperative")
    ownership_choice = input("Choose (1-4): ")
    ownerships = {
        '1': 'Sole Proprietorship',
        '2': 'Partnership', 
        '3': 'Limited Company',
        '4': 'Cooperative'
    }
    business_data['Ownership_Type'] = ownerships.get(ownership_choice, 'Limited Company')
    
    # Location
    print("\n📍 Location:")
    locations = ['Kigali', 'Eastern', 'Northern', 'Southern', 'Western', 'Rwamagana', 'Rusizi', 'Rubavu', 'Huye']
    for i, location in enumerate(locations, 1):
        print(f"{i}. {location}")
    
    location_choice = int(input("Choose location (1-9): ")) - 1
    business_data['Location'] = locations[location_choice]
    
    # Business type
    print("\n🏢 Business Type:")
    print("1. Micro-enterprise")
    print("2. SME")
    print("3. Startup")
    type_choice = input("Choose (1-3): ")
    types = {'1': 'Micro-enterprise', '2': 'SME', '3': 'Startup'}
    business_data['Business_Type'] = types.get(type_choice, 'SME')
    
    return business_data

def display_prediction_result(business_data, result):
    """Display the prediction result in a nice format"""
    print("\n" + "="*60)
    print("🎯 PREDICTION RESULTS")
    print("="*60)
    
    print(f"\n🏢 Business Summary:")
    print(f"   Sector: {business_data['Business_Sector']}")
    print(f"   Capital: {business_data['Initial_Capital']:,.0f} RWF")
    print(f"   Employees: {business_data['Num_Employees']}")
    print(f"   Age: {business_data['Business_Age']} years")
    print(f"   Location: {business_data['Location']}")
    print(f"   Owner: {business_data['Owner_Gender']}, {business_data['Owner_Age']} years old")
    print(f"   Growth: {business_data['Growth_Indicator']:.1%}")
    
    print(f"\n📊 Prediction Results:")
    print(f"   🎯 Prediction: {result['prediction']}")
    print(f"   📈 Success Probability: {result['success_probability']}%")
    print(f"   🎪 Confidence: {result['confidence']}%")
    print(f"   ⚠️  Risk Level: {result['risk_level']}")
    
    print(f"\n💡 Recommendation:")
    print(f"   {result['recommendation']}")
    
    print(f"\n🎯 Advice:")
    print(f"   {result['advice']}")
    
    print(f"\n⏰ Prediction made at: {result['timestamp']}")

def main():
    """Main interactive testing function"""
    print("🧪 SME SUCCESS PREDICTOR - INTERACTIVE TESTING")
    print("="*60)
    print("Enter your business details to get a success prediction!")
    
    # Load model
    model, model_info = load_model_and_info()
    if model is None:
        return
    
    while True:
        try:
            # Get user input
            business_data = get_user_input()
            
            # Make prediction
            result = predict_sme_success(business_data, model, model_info)
            
            # Display results
            display_prediction_result(business_data, result)
            
            # Ask if user wants to test another business
            print("\n" + "="*60)
            continue_choice = input("Would you like to test another business? (y/n): ").lower()
            if continue_choice not in ['y', 'yes']:
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with valid inputs.")
            continue
    
    print("\n✅ Testing session completed!")
    print("🚀 Thank you for using SME Success Predictor!")

if __name__ == "__main__":
    main()