class CropInfo:
    def __init__(self):
        self.crop_database = {
            'rice': {
                'category': 'Cereal',
                'season': 'Kharif (Monsoon)',
                'water_requirement': 'High',
                'soil_type': 'Clay, Clay-loam',
                'climate': 'Tropical, Subtropical',
                'harvest_time': '3-6 months',
                'tips': [
                    'Requires flooded fields during growing season',
                    'Plant in well-puddled soil',
                    'Maintain water depth of 2-5 cm',
                    'Apply nitrogen in split doses'
                ]
            },
            'maize': {
                'category': 'Cereal',
                'season': 'Kharif & Rabi',
                'water_requirement': 'Moderate',
                'soil_type': 'Well-drained loamy soil',
                'climate': 'Warm temperate',
                'harvest_time': '3-4 months',
                'tips': [
                    'Ensure good drainage to prevent waterlogging',
                    'Plant when soil temperature reaches 16°C',
                    'Apply balanced fertilizers',
                    'Control weeds during early growth'
                ]
            },
            'chickpea': {
                'category': 'Legume',
                'season': 'Rabi (Winter)',
                'water_requirement': 'Low to Moderate',
                'soil_type': 'Well-drained sandy loam',
                'climate': 'Cool, dry climate',
                'harvest_time': '3-4 months',
                'tips': [
                    'Avoid waterlogged conditions',
                    'Inoculate seeds with Rhizobium',
                    'Sow after temperature drops below 30°C',
                    'Minimal nitrogen fertilizer needed'
                ]
            },
            'kidneybeans': {
                'category': 'Legume',
                'season': 'Kharif & Rabi',
                'water_requirement': 'Moderate',
                'soil_type': 'Well-drained loamy soil',
                'climate': 'Cool, humid climate',
                'harvest_time': '3-4 months',
                'tips': [
                    'Provide support for climbing varieties',
                    'Ensure good air circulation',
                    'Avoid overhead watering',
                    'Harvest when pods are fully mature'
                ]
            },
            'pigeonpeas': {
                'category': 'Legume',
                'season': 'Kharif',
                'water_requirement': 'Low to Moderate',
                'soil_type': 'Well-drained red soil',
                'climate': 'Tropical, Semi-arid',
                'harvest_time': '4-6 months',
                'tips': [
                    'Drought tolerant crop',
                    'Intercrop with cereals for better yield',
                    'Prune for better branching',
                    'Harvest pods when fully dry'
                ]
            },
            'mothbeans': {
                'category': 'Legume',
                'season': 'Kharif',
                'water_requirement': 'Very Low',
                'soil_type': 'Sandy, well-drained soil',
                'climate': 'Arid, Semi-arid',
                'harvest_time': '2-3 months',
                'tips': [
                    'Extremely drought tolerant',
                    'Suitable for marginal lands',
                    'Minimal input requirements',
                    'Good soil improvement crop'
                ]
            },
            'mungbean': {
                'category': 'Legume',
                'season': 'Kharif & Summer',
                'water_requirement': 'Moderate',
                'soil_type': 'Well-drained loamy soil',
                'climate': 'Warm, humid climate',
                'harvest_time': '2-3 months',
                'tips': [
                    'Short duration crop',
                    'Suitable for multiple cropping',
                    'Harvest pods when they turn brown',
                    'Good nitrogen fixer'
                ]
            },
            'blackgram': {
                'category': 'Legume',
                'season': 'Kharif & Rabi',
                'water_requirement': 'Moderate',
                'soil_type': 'Well-drained black soil',
                'climate': 'Warm, humid climate',
                'harvest_time': '2-3 months',
                'tips': [
                    'Tolerates moderate drought',
                    'Suitable for intercropping',
                    'Harvest when pods turn black',
                    'Store in moisture-proof containers'
                ]
            },
            'lentil': {
                'category': 'Legume',
                'season': 'Rabi',
                'water_requirement': 'Low',
                'soil_type': 'Well-drained sandy loam',
                'climate': 'Cool, dry climate',
                'harvest_time': '3-4 months',
                'tips': [
                    'Cold tolerant crop',
                    'Avoid excessive moisture',
                    'Harvest when pods are dry',
                    'Good rotation crop'
                ]
            },
            'pomegranate': {
                'category': 'Fruit',
                'season': 'Year-round',
                'water_requirement': 'Moderate',
                'soil_type': 'Well-drained loamy soil',
                'climate': 'Semi-arid to temperate',
                'harvest_time': '6-7 months from flowering',
                'tips': [
                    'Prune regularly for better fruit quality',
                    'Provide adequate support to branches',
                    'Mulch around plants',
                    'Harvest when fruits make metallic sound when tapped'
                ]
            },
            'banana': {
                'category': 'Fruit',
                'season': 'Year-round',
                'water_requirement': 'High',
                'soil_type': 'Rich, well-drained soil',
                'climate': 'Tropical, humid climate',
                'harvest_time': '12-15 months',
                'tips': [
                    'Requires consistent moisture',
                    'Protect from strong winds',
                    'Apply organic matter regularly',
                    'Remove suckers for better fruit development'
                ]
            },
            'mango': {
                'category': 'Fruit',
                'season': 'Year-round plantation',
                'water_requirement': 'Moderate',
                'soil_type': 'Well-drained deep soil',
                'climate': 'Tropical, subtropical',
                'harvest_time': '3-5 months from flowering',
                'tips': [
                    'Avoid waterlogging during flowering',
                    'Prune for better light penetration',
                    'Apply organic fertilizers',
                    'Harvest when fruits develop characteristic aroma'
                ]
            },
            'grapes': {
                'category': 'Fruit',
                'season': 'Year-round',
                'water_requirement': 'Moderate',
                'soil_type': 'Well-drained sandy loam',
                'climate': 'Mediterranean, temperate',
                'harvest_time': '4-5 months from pruning',
                'tips': [
                    'Provide proper trellising system',
                    'Prune during dormant season',
                    'Control fungal diseases',
                    'Harvest when sugar content is optimal'
                ]
            },
            'watermelon': {
                'category': 'Fruit',
                'season': 'Summer',
                'water_requirement': 'High',
                'soil_type': 'Sandy loam, well-drained',
                'climate': 'Warm, dry climate',
                'harvest_time': '2-3 months',
                'tips': [
                    'Requires plenty of space to spread',
                    'Deep watering less frequently',
                    'Mulch to retain moisture',
                    'Check for hollow sound when ripe'
                ]
            },
            'muskmelon': {
                'category': 'Fruit',
                'season': 'Summer',
                'water_requirement': 'High',
                'soil_type': 'Sandy loam, well-drained',
                'climate': 'Warm, dry climate',
                'harvest_time': '2-3 months',
                'tips': [
                    'Similar care to watermelon',
                    'Check for sweet aroma when ripe',
                    'Reduce watering near harvest',
                    'Harvest when stem separates easily'
                ]
            },
            'apple': {
                'category': 'Fruit',
                'season': 'Year-round plantation',
                'water_requirement': 'Moderate to High',
                'soil_type': 'Well-drained loamy soil',
                'climate': 'Temperate, cool climate',
                'harvest_time': '4-6 months from flowering',
                'tips': [
                    'Requires winter chilling hours',
                    'Prune for better fruit quality',
                    'Thin fruits for larger size',
                    'Store in cool, humid conditions'
                ]
            },
            'orange': {
                'category': 'Fruit',
                'season': 'Year-round',
                'water_requirement': 'Moderate',
                'soil_type': 'Well-drained citrus soil',
                'climate': 'Subtropical, Mediterranean',
                'harvest_time': '6-8 months from flowering',
                'tips': [
                    'Protect from frost',
                    'Apply citrus-specific fertilizers',
                    'Maintain proper spacing',
                    'Harvest when fully colored'
                ]
            },
            'papaya': {
                'category': 'Fruit',
                'season': 'Year-round',
                'water_requirement': 'High',
                'soil_type': 'Well-drained rich soil',
                'climate': 'Tropical, warm climate',
                'harvest_time': '6-12 months',
                'tips': [
                    'Requires good drainage',
                    'Protect from strong winds',
                    'Apply organic matter regularly',
                    'Harvest when fruits turn yellow'
                ]
            },
            'coconut': {
                'category': 'Fruit/Plantation',
                'season': 'Year-round',
                'water_requirement': 'High',
                'soil_type': 'Sandy, coastal soil',
                'climate': 'Tropical coastal',
                'harvest_time': '12 months',
                'tips': [
                    'Tolerates saline conditions',
                    'Requires consistent moisture',
                    'Apply coconut-specific fertilizers',
                    'Harvest nuts when mature'
                ]
            },
            'cotton': {
                'category': 'Cash Crop',
                'season': 'Kharif',
                'water_requirement': 'Moderate to High',
                'soil_type': 'Black cotton soil',
                'climate': 'Warm, humid climate',
                'harvest_time': '4-6 months',
                'tips': [
                    'Requires warm weather during growth',
                    'Monitor for bollworm infestations',
                    'Ensure adequate moisture during flowering',
                    'Harvest when bolls open completely'
                ]
            },
            'jute': {
                'category': 'Fiber Crop',
                'season': 'Kharif',
                'water_requirement': 'High',
                'soil_type': 'Alluvial, clayey soil',
                'climate': 'Warm, humid climate',
                'harvest_time': '3-4 months',
                'tips': [
                    'Requires high humidity',
                    'Harvest before flowering for quality fiber',
                    'Ret in stagnant water',
                    'Process immediately after harvest'
                ]
            },
            'coffee': {
                'category': 'Plantation Crop',
                'season': 'Year-round',
                'water_requirement': 'High',
                'soil_type': 'Well-drained acidic soil',
                'climate': 'Tropical highland',
                'harvest_time': '6-8 months from flowering',
                'tips': [
                    'Requires shade in hot climates',
                    'Maintain soil pH between 6.0-6.5',
                    'Apply organic mulch regularly',
                    'Harvest when berries are fully red'
                ]
            }
        }
    
    def get_crop_info(self, crop_name):
        """Get information for a specific crop"""
        return self.crop_database.get(crop_name.lower(), None)
    
    def get_all_crops(self):
        """Get list of all supported crops"""
        return list(self.crop_database.keys())
    
    def get_crops_by_category(self, category):
        """Get crops by category"""
        return [crop for crop, info in self.crop_database.items() 
                if info['category'].lower() == category.lower()]
