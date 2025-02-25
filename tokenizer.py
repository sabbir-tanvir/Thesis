class BnGraphemizer:
    """
    A class for Bengali (Bangla) text tokenization at the grapheme level.
    Breaks down Bengali text into individual graphemes including:
    - Consonants
    - Vowels
    - Diacritics
    - Special characters
    - Conjuncts (handles these as separate units when needed)
    """
    
    def __init__(self):
        # Bengali vowels (স্বরবর্ণ)
        self.vowels = set([
            'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ'
        ])
        
        # Bengali consonants (ব্যঞ্জনবর্ণ)
        self.consonants = set([
            'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 
            'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 
            'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', 'ড়', 'ঢ়', 'য়', 'ৎ'
        ])
        
        # Vowel diacritics (কার)
        self.diacritics = set([
            'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ', '্'
        ])
        
        # Bengali special characters
        self.special_chars = set([
            'ং', 'ঃ', 'ঁ', '।', '৷', '॥'
        ])
        
        # Bengali numerals (সংখ্যা)
        self.numerals = set([
            '০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯'
        ])
        
        # Common conjuncts (যুক্তাক্ষর) - these will be treated as single tokens when needed
        self.common_conjuncts = set([
            'ক্ষ', 'জ্ঞ', 'শ্র', 'ত্র', 'দ্ব', 'দ্ধ', 'দ্ম', 'ন্ড', 'ন্ঠ', 
            'ন্ত', 'ন্থ', 'ন্দ', 'ন্ধ', 'ম্প', 'ম্ফ', 'ষ্ণ', 'হ্ন'
        ])
    
    def tokenize(self, text):
        """
        Tokenize Bengali text into individual graphemes.
        
        Args:
            text (str): Bengali text input
            
        Returns:
            list: List of individual graphemes
        """
        tokens = []
        i = 0
        
        while i < len(text):
            # Check for conjuncts (look ahead for hasant)
            if i + 2 < len(text) and text[i] in self.consonants and text[i+1] == '্' and text[i+2] in self.consonants:
                # Check if this is a common conjunct (up to 3-char sequence)
                conjunct = text[i:i+3]
                if conjunct in self.common_conjuncts:
                    tokens.append(conjunct)
                    i += 3
                    continue
                else:
                    # If not a common conjunct, tokenize normally
                    tokens.append(text[i])
                    i += 1
                    continue
            
            # Regular character
            tokens.append(text[i])
            i += 1
        
        return tokens
    
    def is_vowel(self, char):
        """Check if a character is a Bengali vowel."""
        return char in self.vowels
    
    def is_consonant(self, char):
        """Check if a character is a Bengali consonant."""
        return char in self.consonants
    
    def is_diacritic(self, char):
        """Check if a character is a Bengali diacritic."""
        return char in self.diacritics
    
    def is_special_char(self, char):
        """Check if a character is a Bengali special character."""
        return char in self.special_chars
    
    def is_numeral(self, char):
        """Check if a character is a Bengali numeral."""
        return char in self.numerals


# Example usage
if __name__ == "__main__":
    tokenizer = BnGraphemizer()
    
    # Test cases
    test_cases = [
        "বাংলা",  # Bangla
        "আমি বাংলায় কথা বলি",  # I speak Bengali
        "ক্ষমতা",  # Contains a conjunct
        "১২৩৪৫",  # Bengali numerals
    ]
    
    for test in test_cases:
        tokens = tokenizer.tokenize(test)
        print(f"Text: {test}")
        print(f"Tokens: {tokens}")
        print("-" * 30)