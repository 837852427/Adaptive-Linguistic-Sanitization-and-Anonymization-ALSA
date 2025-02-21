import PLRS
import CIIS
import TRS
import CASM

class ALSA:
    """
    ALSA class to calculate the ALSA metrics for each word.
    """
    def __init__(self, model, tokenizer, csv_path, model_name = "gpt2"):
        """
        Initialize the ALSA class.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.csv_path = csv_path
        self.PLRS = PLRS.PLRS()
        self.CIIS = CIIS.CIIS(model, tokenizer)
        self.TRS = TRS.TRS(model, tokenizer)
        self.CASM = CASM.CASM() 
    
    def calculate(self):
        """
        Calculate the ALSA metrics for each word.
        """
        triple_metrics = self.calculate_part1()
        self.calculate_part2(triple_metrics)

        print('\n\033[1;32mAll completed\033[0m')

    
    def calculate_part1(self):
        """
        Calculate the ALSA metrics for each word.
        """
        # Calculate starting
        print('\033[1;32mCalculating part 1...\033[0m')

        # Step 1: Calculate the PLRS metrics
        print('\n\033[1mCalculating PLRS metrics...\033[0m')

        PLRS_metrics = self.PLRS.calculate_plrs()

        print('\033[1;32mPLRS calculation completed\033[0m')
        
        # Step 2: Calculate the CIIS metrics
        print('\n\033[1mCalculating CIIS metrics...\033[0m')

        CIIS_metrics = self.CIIS.calculate(self.csv_path)

        print('\033[1;32mCIIS calculation completed\033[0m')
        
        # Step 3: Calculate the TRS metrics
        print('\n\033[1mCalculating TRS metrics...\033[0m')

        TRS_metrics = self.TRS.calculate(self.csv_path)

        print('\033[1;32mTRS calculation completed\033[0m')
        
        # Step 4: Calculate the CASM metrics
        print('\n\033[1mCalculating CASM metrics...\033[0m')
        CASM_metrics = {}
        for (word, score) in PLRS_metrics.items():
            CASM_metrics[word] = [score]
        
        for (word, score) in CIIS_metrics.items():
            if word in CASM_metrics:
                CASM_metrics[word].append(score)\
            # 这里或许是应该修改
            else:   
                CASM_metrics[word] = [0, score]
        
        for (word, score) in TRS_metrics.items():
            if word in CASM_metrics:
                CASM_metrics[word].append(score)
            else:
                CASM_metrics[word] = [0, 0, score]
        
        print('\033[1mCASM calculation completed\033[0m')

        print('\n\033[1;32mPart 1 calculation completed\033[0m')

        return CASM_metrics
    
    def calculate_part2(self, triple_metrics):
        """
        Calculate the ALSA metrics for each word.
        """
        print('\n\033[1;32mCalculating part 2\033[0m')
        self.CASM.calculate_casm(triple_metrics)
        print('\n\033[1;32mPart 2 calculation completed\033[0m')


    