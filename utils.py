import numpy as np

def validate_input(input_features):
    """Validate user input parameters"""
    if len(input_features) != 7:
        return {'valid': False, 'message': 'Please provide all 7 parameters'}
    
    # Define reasonable ranges for each parameter
    ranges = get_feature_ranges()
    feature_names = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    
    for i, (value, name) in enumerate(zip(input_features, feature_names)):
        param_name = name.lower()
        if param_name == 'temperature':
            param_name = 'temperature'
        elif param_name == 'humidity':
            param_name = 'humidity'
        elif param_name == 'ph':
            param_name = 'ph'
        elif param_name == 'rainfall':
            param_name = 'rainfall'
        else:
            param_name = name
        
        min_val = ranges[param_name]['min']
        max_val = ranges[param_name]['max']
        
        if not (min_val <= value <= max_val):
            return {
                'valid': False, 
                'message': f'{name} should be between {min_val} and {max_val}'
            }
    
    return {'valid': True, 'message': 'All inputs are valid'}

def get_feature_ranges():
    """Get valid ranges for input features"""
    return {
        'N': {'min': 0, 'max': 200, 'default': 50},
        'P': {'min': 0, 'max': 150, 'default': 50},
        'K': {'min': 0, 'max': 250, 'default': 50},
        'temperature': {'min': 0, 'max': 50, 'default': 25},
        'humidity': {'min': 0, 'max': 100, 'default': 50},
        'ph': {'min': 3, 'max': 10, 'default': 6.5},
        'rainfall': {'min': 0, 'max': 300, 'default': 100}
    }

def format_prediction_result(crop_name, confidence):
    """Format prediction result for display"""
    confidence_level = "High" if confidence >= 80 else "Medium" if confidence >= 60 else "Low"
    
    return {
        'crop': crop_name.title(),
        'confidence': round(confidence, 2),
        'confidence_level': confidence_level,
        'recommendation': get_recommendation_text(confidence_level)
    }

def get_recommendation_text(confidence_level):
    """Get recommendation text based on confidence level"""
    recommendations = {
        'High': "This crop is an excellent match for your conditions. Proceed with confidence!",
        'Medium': "This crop is a good match. Consider local conditions and market demand.",
        'Low': "Consider this crop carefully. You may want to explore alternatives or adjust conditions."
    }
    return recommendations.get(confidence_level, "")

def calculate_nutrient_balance(n, p, k):
    """Calculate nutrient balance ratio"""
    total = n + p + k
    if total == 0:
        return {'N': 0, 'P': 0, 'K': 0}
    
    return {
        'N': round((n / total) * 100, 1),
        'P': round((p / total) * 100, 1),
        'K': round((k / total) * 100, 1)
    }

def get_environmental_assessment(temperature, humidity, rainfall):
    """Assess environmental conditions"""
    assessments = []
    
    # Temperature assessment
    if temperature < 15:
        assessments.append("Cool temperature - suitable for temperate crops")
    elif temperature > 35:
        assessments.append("High temperature - choose heat-tolerant crops")
    else:
        assessments.append("Moderate temperature - suitable for most crops")
    
    # Humidity assessment
    if humidity < 30:
        assessments.append("Low humidity - consider drought-tolerant crops")
    elif humidity > 80:
        assessments.append("High humidity - watch for fungal diseases")
    else:
        assessments.append("Moderate humidity - good for most crops")
    
    # Rainfall assessment
    if rainfall < 50:
        assessments.append("Low rainfall - irrigation may be needed")
    elif rainfall > 200:
        assessments.append("High rainfall - ensure good drainage")
    else:
        assessments.append("Adequate rainfall for most crops")
    
    return assessments
