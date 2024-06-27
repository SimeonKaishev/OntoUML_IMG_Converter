import re

class TextUtils:
    """
    A utility class for processing text related to UML class diagrams.

    Methods
    -------
    process_class_text(text)
        Processes the text of a UML class diagram to replace special characters and fix formatting issues.
    
    extract_class_info(text)
        Extracts the stereotype and class name from the processed UML class text.
    """

    @staticmethod
    def process_class_text(text):
        """
        Processes the text of a UML class diagram to replace special characters and fix formatting issues.

        Parameters
        ----------
        text : str
            The text of the UML class diagram.

        Returns
        -------
        str
            The processed text with special characters replaced and formatting issues fixed.
        """
        # Replace special characters
        text = text.replace("«", "<<")
        text = text.replace("< <", "<<")
        text = text.replace("»", ">>")
        text = text.replace("::", ":")
        
        # Shouldn't have more than 1 stereotype
        if text.count("<<") > 1:
            return ""

        lines = text.split('\n')
        if not lines:
            return text

        # Check if the second line includes '<<' or '<'
        if len(lines) > 1 and ('>>' in lines[1] or '>' in lines[1]):
            # Combine the first and second lines
            lines[0] = lines[0] + ' ' + lines[1]
            # Remove the second line from the list
            lines.pop(1)
        
        first_line = lines[0]    

        # Check if the first line contains '<<' and does not contain '>>'
        if '<<' in first_line and '>>' not in first_line:
            # Check if the last character of the first line is 's'
            if first_line.endswith('s'):
                # Replace the last 's' with '>>'
                first_line = first_line[:-1] + '>>'
            elif first_line.endswith('>s'):
                # Replace the last 's' with '>>'
                first_line = first_line[:-1] + '>'
            elif first_line.endswith('> >'):
                # Replace the last 's' with '>>'
                first_line = first_line[:-2] + '>'

        # Update the first line in the list of lines
        lines[0] = first_line
        
        # Join the lines back into a single string
        return '\n'.join(lines)

    @staticmethod
    def extract_class_info(text):
        """
        Extracts the stereotype and class name from the processed UML class text.

        Parameters
        ----------
        text : str
            The text of the UML class diagram.

        Returns
        -------
        tuple
            A tuple containing the stereotype and the class name.
        """
        text = TextUtils.process_class_text(text)

        # Extract the stereotype
        pattern = r"<<(.*?)>>"
        match = re.search(pattern, text)
        if match:
            stereotype = match.group(1)
        else:
            print("No stereotype found")
            stereotype = ""

        # Extract the class name
        if stereotype != "":
            # Split the input string into lines and ignore the first line containing <<>>
            lines = text.split('\n')[1:]
            text = '\n'.join(lines).strip()

        # Remove new lines
        text = text.replace('\n', ' ')

        # Split the remaining text based on the colon, to remove package name
        parts = text.split(':')
        if len(parts) > 1:
            before_colon = parts[0].strip()
            after_colon = parts[1].strip()
            return stereotype, after_colon
        else:
            before_colon = text.strip()
            return stereotype, before_colon