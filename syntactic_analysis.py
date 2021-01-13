import json
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Token, Span
from spacy.pipeline import EntityRuler
# from spacy.strings import StringStore
import os
from flask import Flask, globals, Response, request

app = Flask(__name__)


class ClausesComponent(object):
    """spaCy v2.0 pipeline component that requests all Escape Clauses via
    the REST EscapeClauses API, merges EscapeClauses names into one token, assigns entity
    labels and sets attributes on EscapeClauses tokens.
    """

    name = "clauses"  # component name, will show up in the pipeline

    def __init__(self, nlp, label="Clause"):
        """Initialise the pipeline component. The shared nlp instance is used
        to initialise the matcher with the shared vocab, get the label ID and
        generate Doc objects as phrase match patterns.
        """
        # Make request once on initialisation and store the data
        # r = requests.get("https://restcountries.eu/rest/v2/all")
        # r.raise_for_status()  # make sure requests raises an error if it fails
        # countries = r.json()
        self.labels = ("ESCAPE_CLAUSE", "OPEN_CLAUSE")
        # Convert API response to dict keyed by country name for easy lookup
        # This could also be extended using the alternative and foreign language
        # names provided by the API
        # self.countries = {c["name"]: c for c in countries}
        self.escapeclauses = [
            "so far as is possible", "as little as possible", "where possible",
            "as much as possible", "if it should prove necessary",
            "if necessary", "to the extent necessary", "as appropriate",
            "as required", "to the extent practical", "if practicable"
        ]
        self.openclauses = [
            "including but not limited to", "etc.", "et cetera", "and so on"
        ]
        #         self.label = nlp.vocab.strings[label]  # get entity label ID

        # Set up the PhraseMatcher with Doc patterns for each country name
        # patterns = [nlp(c) for c in self.countries.keys()]

        # Only run nlp.make_doc to speed things up
        escapeclauses_patterns = [
            nlp.make_doc(text) for text in self.escapeclauses
        ]
        openclauses_patterns = [
            nlp.make_doc(text) for text in self.openclauses
        ]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add("ESCAPE_CLAUSE", None, *escapeclauses_patterns)
        self.matcher.add("OPEN_CLAUSE", None, *openclauses_patterns)

        # Register attribute on the Token. We'll be overwriting this based on
        # the matches, so we're only setting a default value, not a getter.
        # If no default value is set, it defaults to None.
        Token.set_extension("is_escapeclause", default=False, force=True)
        Token.set_extension("is_openclause", default=False, force=True)

        # Register attributes on Doc and Span via a getter that checks if one of
        # the contained tokens is set to is_country == True.
        Doc.set_extension("has_escapeclause",
                          getter=self.has_escapeclause,
                          force=True)
        Span.set_extension("has_escapeclause",
                           getter=self.has_escapeclause,
                           force=True)
        Doc.set_extension("has_openclause",
                          getter=self.has_openclause,
                          force=True)
        Span.set_extension("has_openclause",
                           getter=self.has_openclause,
                           force=True)

    def __call__(self, doc):
        """Apply the pipeline component on a Doc object and modify it if matches
        are found. Return the Doc, so it can be processed by the next component
        in the pipeline, if available.
        """
        matches = self.matcher(doc)
        spans = []  # keep the spans for later so we can merge them afterwards
        for match_id, start, end in matches:
            # Generate Span representing the entity & set label
            #             entity = Span(doc, start, end, label=self.label)
            label = doc.vocab.strings[match_id]
            entity = Span(doc, start, end, label=label)
            spans.append(entity)
            # Set custom attribute on each token of the entity
            # Can be extended with other data returned by the API, like
            # currencies, country code, flag, calling code etc.
            for token in entity:
                if label == "ESCAPE_CLAUSE":
                    token._.set("is_escapeclause", True)
                    token.ent_id_ = 'R8'
                if label == "OPEN_CLAUSE":
                    token._.set("is_openclause", True)
                    token.ent_id_ = 'R9'
            # Overwrite doc.ents and add entity – be careful not to replace!
            doc.ents = list(doc.ents) + [entity]


#         for span in spans:
#             # Iterate over all spans and merge them into one token. This is done
#             # after setting the entities – otherwise, it would cause mismatched
#             # indices!
#             span.merge()

        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span)

        return doc  # don't forget to return the Doc!

    def has_escapeclause(self, tokens):
        """Getter for Doc and Span attributes. Returns True if one of the tokens
        is a escapeclause. Since the getter is only called when we access the
        attribute, we can refer to the Token's 'is_escapeclause' attribute here,
        which is already set in the processing step."""
        return any([t._.get("is_escapeclause") for t in tokens])

    def has_openclause(self, tokens):
        """Getter for Doc and Span attributes. Returns True if one of the tokens
        is a escapeclause. Since the getter is only called when we access the
        attribute, we can refer to the Token's 'is_openclause' attribute here,
        which is already set in the processing step."""
        return any([t._.get("is_openclause") for t in tokens])


class CombinatorComponent(object):
    """spaCy v2.0 pipeline component that requests all Combinators via
    the REST Combinators API, merges Combinators names into one token, assigns entity
    labels and sets attributes on Combinators tokens.
    """

    name = "combinators"  # component name, will show up in the pipeline

    def __init__(self, nlp, label="Combinators"):
        """Initialise the pipeline component. The shared nlp instance is used
        to initialise the matcher with the shared vocab, get the label ID and
        generate Doc objects as phrase match patterns.
        """
        self.labels = ("COMBINATORS", )
        self.combinators = [
            "and", "or", "then", "unless", "but", "as well as", "however",
            "whether", "meanwhile", "whereas", "on the other hand", "otherwise"
        ]

        #         self.label = nlp.vocab.strings[label]  # get entity label ID

        # Only run nlp.make_doc to speed things up
        combinators_patterns = [
            nlp.make_doc(text) for text in self.combinators
        ]

        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add("COMBINATORS", None, *combinators_patterns)

        # Register attribute on the Token. We'll be overwriting this based on
        # the matches, so we're only setting a default value, not a getter.
        # If no default value is set, it defaults to None.
        Token.set_extension("is_combinator", default=False, force=True)

        # Register attributes on Doc and Span via a getter that checks if one of
        # the contained tokens is set to is_country == True.
        Doc.set_extension("has_combinator",
                          getter=self.has_combinator,
                          force=True)
        Span.set_extension("has_combinator",
                           getter=self.has_combinator,
                           force=True)

    def __call__(self, doc):
        """Apply the pipeline component on a Doc object and modify it if matches
        are found. Return the Doc, so it can be processed by the next component
        in the pipeline, if available.
        """
        matches = self.matcher(doc)
        spans = []  # keep the spans for later so we can merge them afterwards
        # get head token
        headtoken = ''
        for token in doc:
            if token.dep_ == 'ROOT':
                headtoken = token

        if headtoken:
            for match_id, start, end in matches:
                # Generate Span representing the entity & set label
                #             entity = Span(doc, start, end, label=self.label)
                label = doc.vocab.strings[match_id]
                entity = Span(doc, start, end, label=label)
                # mark if the flag is_combinator is set. default to False
                is_combinator_set = False
                spans.append(entity)
                # Set is_combinator attribute on each token of the entity
                # When the combinator is combining sub-clauses
                for token in entity:
                    if label == "COMBINATORS" and token.head == headtoken:
                        token._.set("is_combinator", True)
                        token.ent_id_ = 'R19'
                        is_combinator_set = True
                if is_combinator_set:
                    # Overwrite doc.ents and add entity – be careful not to replace!
                    doc.ents = list(doc.ents) + [entity]

        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span)

        return doc  # don't forget to return the Doc!

    def has_combinator(self, tokens):
        """Getter for Doc and Span attributes. Returns True if one of the tokens
        is a combinator. Since the getter is only called when we access the
        attribute, we can refer to the Token's 'is_combinator' attribute here,
        which is already set in the processing step."""
        return any([t._.get("is_combinator") for t in tokens])


# Validate input data is JSON
def is_json(data):
    try:
        json_object = json.loads(data)
    except ValueError as e:
        return False
    return True


# Define the getter function
def get_R1_is_complete(doc):
    # Return if the 3 parts exist: ROOT, nsubj, dobj
    headtoken = ''
    for token in doc:
        if token.dep_ == 'ROOT':
            headtoken = token
    if headtoken:
        return any('subj' in token.dep_ for token in headtoken.subtree) \
        and any('obj' in token.dep_ for token in headtoken.subtree)
    else:
        #       'no ROOT found'
        return False


def get_R2_no_passive(doc):
    headtoken = ''
    for token in doc:
        if token.dep_ == 'ROOT':
            headtoken = token
    if headtoken:
        return not (any(('pass' in token.dep_) and (token.head == headtoken)
                        for token in doc))
    else:
        return 'no ROOT found'


def get_R5_definite_article(doc):
    # return not(any(token.lemma_ in ('a','an') and token.dep_ == 'det' for token in doc))
    return not (any(token.ent_type_ == 'INDEFINITE_ARTICLE' for token in doc))


def get_R6_has_num_quans(doc):
    # this check need NER pipe to work
    # return any(token.ent_type_ in ('CARDINAL','QUANTITY','TIME', 'DATE') for token in doc)
    # Return if any of the tokens in the doc return True for token.like_num
    return any(token.like_num for token in doc)


def get_R7_no_vague_terms(doc):
    # return not(any(token.lemma_ in('some','any','allowable','several','many','few','almost','nearly',
    #                           'lot','about','around', 'close','approximate','up')
    #           and token.dep_ in('advmod','amod','quantmod')  for token in doc))
    return not (any(token.ent_type_ == 'VAGUE_TERM' for token in doc))


def get_R10_no_superfluous(doc):
    return not (any(token.ent_type_ == 'SUPERFLUOUS' for token in doc))


def get_R16_no_not(doc):
    return not (any(token.ent_type_ == 'NOT' for token in doc))


def get_R17_no_oblique(doc):
    return not (any(token.text == '/' and token.nbor(-2).pos_ != 'NUM'
                    for token in doc))


def get_R18_single_sent(doc):
    return len(list(doc.sents)) == 1


def get_R20_no_purpose(doc):
    return not (any(token.ent_type_ == 'PURPOSE' for token in doc))


def get_R21_no_parenthese(doc):
    return not (any(token.ent_type_ == 'PARENTHESE' for token in doc))


def get_R24_no_pronoun(doc):
    return not (any(token.ent_type_ == 'PRONOUN' for token in doc))


def get_R26_no_unachievable(doc):
    return not (any(token.ent_type_ == 'UNACHIEVABLE' for token in doc))


def get_R35_no_indefinite_temporal(doc):
    return not (any(token.ent_type_ == 'INDEFINITE_TEMPORAL' for token in doc))


# Register the Doc property extension "R1_is_complete" with the getter get_R1_is_complete
Doc.set_extension("R1_is_complete", getter=get_R1_is_complete)
# Register the Doc property extension "R2_no_passive" with the getter get_R2_no_passive
Doc.set_extension("R2_no_passive", getter=get_R2_no_passive)
# Register the Doc property extension "R5_definite_article" with the getter get_R5_definite_article
Doc.set_extension("R5_definite_article", getter=get_R5_definite_article)
# Register the Doc property extension "R6_has_num_quans" with the getter get_R6_has_num_quans
Doc.set_extension("R6_has_num_quans", getter=get_R6_has_num_quans)
# Register the Doc property extension "R7_no_vague_terms" with the getter get_R7_no_vague_terms
Doc.set_extension("R7_no_vague_terms", getter=get_R7_no_vague_terms)
# Register the Doc property extension "R10_no_superfluous" with the getter get_R7_no_vague_terms
Doc.set_extension("R10_no_superfluous", getter=get_R10_no_superfluous)
# Register the Doc property extension "R16_no_not" with the getter get_R16_no_not
Doc.set_extension("R16_no_not", getter=get_R16_no_not, force=True)
# Register the Doc property extension "R17_no_oblique" with the getter get_R17_no_oblique
Doc.set_extension("R17_no_oblique", getter=get_R17_no_oblique, force=True)
# Register the Doc property extension "R18_single_sent" with the getter get_R18_single_sent
Doc.set_extension("R18_single_sent", getter=get_R18_single_sent, force=True)
# Register the Doc property extension "R20_no_purpose" with the getter get_R20_no_purpose
Doc.set_extension("R20_no_purpose", getter=get_R20_no_purpose, force=True)
# Register the Doc property extension "R21_no_parenthese" with the getter get_R21_no_parenthese
Doc.set_extension("R21_no_parenthese",
                  getter=get_R21_no_parenthese,
                  force=True)
# Register the Doc property extension "R24_no_pronoun" with the getter get_R24_no_pronoun
Doc.set_extension("R24_no_pronoun", getter=get_R24_no_pronoun, force=True)
# Register the Doc property extension "R26_no_unachievable" with the getter get_R26_no_unachievable
Doc.set_extension("R26_no_unachievable",
                  getter=get_R26_no_unachievable,
                  force=True)
# Register the Doc property extension "R35_no_indefinite_temporal" with the getter get_R35_no_indefinite_temporal
Doc.set_extension("R35_no_indefinite_temporal",
                  getter=get_R35_no_indefinite_temporal,
                  force=True)


# Validate requirement statements
def validate(texts):
    results = []
    success = False
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])

        nlp.add_pipe(ClausesComponent(nlp), before="tagger")
        nlp.add_pipe(CombinatorComponent(nlp), after="parser")

        ruler = EntityRuler(nlp)

        patterns = [
            # patterns for rule R5
            {
                "label": "INDEFINITE_ARTICLE",
                "pattern": [{
                    'LOWER': {
                        'IN': ['a', 'an']
                    },
                    'DEP': 'det'
                }],
                "id": "R5"
            },
            # patterns for rule R7
            {
                "label": "VAGUE_TERM",
                "pattern": [{
                    'LOWER': 'some'
                }],
                "id": "R7"
            },
            {
                "label": "VAGUE_TERM",
                "pattern": [{
                    'LOWER': 'any'
                }],
                "id": "R7"
            },
            {
                "label": "VAGUE_TERM",
                "pattern": [{
                    'LOWER': 'allowable'
                }],
                "id": "R7"
            },
            {
                "label": "VAGUE_TERM",
                "pattern": [{
                    'LOWER': 'several'
                }],
                "id": "R7"
            },
            {
                "label": "VAGUE_TERM",
                "pattern": [{
                    'LOWER': 'many'
                }],
                "id": "R7"
            },
            {
                "label": "VAGUE_TERM",
                "pattern": [{
                    'LOWER': 'few',
                    'POS': 'ADJ'
                }],
                "id": "R7"
            },
            {
                "label": "VAGUE_TERM",
                "pattern": [{
                    'LOWER': 'a'
                }, {
                    'LOWER': 'lot'
                }],
                "id": "R7"
            },
            {
                "label": "VAGUE_TERM",
                "pattern": [{
                    'LOWER': 'about',
                    'POS': 'ADV'
                }],
                "id": "R7"
            },
            {
                "label": "VAGUE_TERM",
                "pattern": [{
                    'LOWER': 'almost'
                }],
                "id": "R7"
            },
            {
                "label": "VAGUE_TERM",
                "pattern": [{
                    'LOWER': 'nearly'
                }],
                "id": "R7"
            },
            {
                "label": "VAGUE_TERM",
                "pattern": [{
                    'LOWER': 'around',
                    'DEP': 'advmod'
                }],
                "id": "R7"
            },
            {
                "label": "VAGUE_TERM",
                "pattern": [{
                    'LOWER': 'close'
                }, {
                    'LOWER': 'to'
                }],
                "id": "R7"
            },
            {
                "label": "VAGUE_TERM",
                "pattern": [{
                    'LOWER': 'approximate'
                }],
                "id": "R7"
            },
            {
                "label": "VAGUE_TERM",
                "pattern": [{
                    'LOWER': 'up'
                }, {
                    'LOWER': 'to'
                }],
                "id": "R7"
            },
            # patterns for rule R10
            {
                "label":
                "SUPERFLUOUS",
                "pattern": [{
                    'LEMMA': 'be'
                }, {
                    'LOWER': 'designed'
                }, {
                    'LOWER': 'to'
                }],
                "id":
                "R10"
            },
            {
                "label": "SUPERFLUOUS",
                "pattern": [{
                    'LEMMA': 'be'
                }, {
                    'LOWER': 'able'
                }, {
                    'LOWER': 'to'
                }],
                "id": "R10"
            },
            {
                "label": "SUPERFLUOUS",
                "pattern": [{
                    'LEMMA': 'be'
                }, {
                    'LOWER': 'capable'
                }, {
                    'LOWER': 'of'
                }],
                "id": "R10"
            },
            # patterns for rule R16
            {
                "label": "NOT",
                "pattern": [{
                    'LEMMA': 'not'
                }],
                "id": "R16"
            },
            # patterns for rule R20
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'purpose'
                }, {
                    'LOWER': 'of'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'purpose'
                }, {
                    'LOWER': 'to'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'in'
                }, {
                    'LOWER': 'order'
                }, {
                    'LOWER': 'to'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'for'
                }, {
                    'LOWER': 'fear'
                }, {
                    'LOWER': 'that'
                }],
                "id": "R20"
            },
            {
                "label":
                "PURPOSE",
                "pattern": [{
                    'LOWER': 'in'
                }, {
                    'LOWER': 'the'
                }, {
                    'LOWER': 'hope'
                }, {
                    'LOWER': 'that'
                }],
                "id":
                "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'so'
                }, {
                    'LOWER': 'that'
                }],
                "id": "R20"
            },
            {
                "label":
                "PURPOSE",
                "pattern": [{
                    'LOWER': 'with'
                }, {
                    'LOWER': 'this'
                }, {
                    'LOWER': 'in'
                }, {
                    'LOWER': 'mind'
                }],
                "id":
                "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'hence'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'therefore'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'thus'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'accordingly'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'thence'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'in'
                }, {
                    'LOWER': 'consequence'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'thereby'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'wherefore'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'then'
                }],
                "id": "R20"
            },
            {
                "label":
                "PURPOSE",
                "pattern": [{
                    'LOWER': 'because'
                }, {
                    'LOWER': 'of'
                }, {
                    'LOWER': 'this'
                }],
                "id":
                "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'henceforth'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'this'
                }, {
                    'LOWER': 'is'
                }, {
                    'LOWER': 'why'
                }],
                "id": "R20"
            },
            {
                "label":
                "PURPOSE",
                "pattern": [{
                    'LOWER': 'that'
                }, {
                    'LOWER': 'being'
                }, {
                    'LOWER': 'the'
                }, {
                    'LOWER': 'case'
                }],
                "id":
                "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'subsequently'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'because'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'whence'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'in'
                }, {
                    'LOWER': 'this'
                }, {
                    'LOWER': 'way'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'thereupon'
                }],
                "id": "R20"
            },
            {
                "label":
                "PURPOSE",
                "pattern": [{
                    'LOWER': 'in'
                }, {
                    'LOWER': 'view'
                }, {
                    'LOWER': 'of'
                }, {
                    'LOWER': 'this'
                }],
                "id":
                "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'that'
                }, {
                    'LOWER': 'is'
                }, {
                    'LOWER': 'why'
                }],
                "id": "R20"
            },
            {
                "label":
                "PURPOSE",
                "pattern": [{
                    'LOWER': 'following'
                }, {
                    'LOWER': 'from'
                }, {
                    'LOWER': 'this'
                }],
                "id":
                "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'therefrom'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'in'
                }, {
                    'LOWER': 'doing'
                }, {
                    'LOWER': 'so'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'in'
                }, {
                    'LOWER': 'that'
                }, {
                    'LOWER': 'case'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'accordingly'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'in'
                }, {
                    'LOWER': 'consequence'
                }],
                "id": "R20"
            },
            {
                "label": "PURPOSE",
                "pattern": [{
                    'LOWER': 'due'
                }, {
                    'LOWER': 'to'
                }],
                "id": "R20"
            },
            {
                "label":
                "PURPOSE",
                "pattern": [{
                    'LOWER': 'as'
                }, {
                    'LOWER': 'a'
                }, {
                    'LOWER': 'consequence'
                }],
                "id":
                "R20"
            },
            # patterns for rule R21
            {
                "label": "PARENTHESE",
                "pattern": [{
                    'LOWER': '('
                }],
                "id": "R21"
            },
            {
                "label": "PARENTHESE",
                "pattern": [{
                    'LOWER': ')'
                }],
                "id": "R21"
            },
            {
                "label": "PARENTHESE",
                "pattern": [{
                    'LOWER': '['
                }],
                "id": "R21"
            },
            {
                "label": "PARENTHESE",
                "pattern": [{
                    'LOWER': ']'
                }],
                "id": "R21"
            },
            # patterns for R24: pronoun
            {
                "label": "PRONOUN",
                "pattern": [{
                    'POS': 'PRON'
                }],
                "id": "R24"
            },
            {
                "label": "PRONOUN",
                "pattern": [{
                    'LEMMA': '-PRON-'
                }],
                "id": "R24"
            },
            # patterns for R24: Indefinite pronouns
            {
                "label": "PRONOUN",
                "pattern": [{
                    'LOWER': 'everyone'
                }],
                "id": "R24"
            },
            {
                "label": "PRONOUN",
                "pattern": [{
                    'LOWER': 'everybody'
                }],
                "id": "R24"
            },
            {
                "label": "PRONOUN",
                "pattern": [{
                    'LOWER': 'everywhere'
                }],
                "id": "R24"
            },
            {
                "label": "PRONOUN",
                "pattern": [{
                    'LOWER': 'everything'
                }],
                "id": "R24"
            },
            {
                "label": "PRONOUN",
                "pattern": [{
                    'LOWER': 'someone'
                }],
                "id": "R24"
            },
            {
                "label": "PRONOUN",
                "pattern": [{
                    'LOWER': 'somebody'
                }],
                "id": "R24"
            },
            {
                "label": "PRONOUN",
                "pattern": [{
                    'LOWER': 'somewhere'
                }],
                "id": "R24"
            },
            {
                "label": "PRONOUN",
                "pattern": [{
                    'LOWER': 'something'
                }],
                "id": "R24"
            },
            {
                "label": "PRONOUN",
                "pattern": [{
                    'LOWER': 'anyone'
                }],
                "id": "R24"
            },
            {
                "label": "PRONOUN",
                "pattern": [{
                    'LOWER': 'anybody'
                }],
                "id": "R24"
            },
            {
                "label": "PRONOUN",
                "pattern": [{
                    'LOWER': 'anywhere'
                }],
                "id": "R24"
            },
            {
                "label": "PRONOUN",
                "pattern": [{
                    'LOWER': 'anything'
                }],
                "id": "R24"
            },
            {
                "label": "PRONOUN",
                "pattern": [{
                    'LOWER': 'no'
                }, {
                    'LOWER': 'one'
                }],
                "id": "R24"
            },
            {
                "label": "PRONOUN",
                "pattern": [{
                    'LOWER': 'nobody'
                }],
                "id": "R24"
            },
            {
                "label": "PRONOUN",
                "pattern": [{
                    'LOWER': 'nowhere'
                }],
                "id": "R24"
            },
            {
                "label": "PRONOUN",
                "pattern": [{
                    'LOWER': 'nothing'
                }],
                "id": "R24"
            },
            # pattern for rule R26
            {
                "label": "UNACHIEVABLE",
                "pattern": [{
                    'LOWER': {
                        'IN': ['100', '0']
                    }
                }, {
                    'LOWER': '%'
                }],
                "id": "R26"
            },
            # pattern for rule R35
            {
                "label": "INDEFINITE_TEMPORAL",
                "pattern": [{
                    'LOWER': 'eventually'
                }],
                "id": "R35"
            },
            {
                "label": "INDEFINITE_TEMPORAL",
                "pattern": [{
                    'LOWER': 'until'
                }],
                "id": "R35"
            },
            {
                "label": "INDEFINITE_TEMPORAL",
                "pattern": [{
                    'LOWER': 'before'
                }],
                "id": "R35"
            },
            {
                "label": "INDEFINITE_TEMPORAL",
                "pattern": [{
                    'LOWER': 'when'
                }],
                "id": "R35"
            },
            {
                "label": "INDEFINITE_TEMPORAL",
                "pattern": [{
                    'LOWER': 'after'
                }],
                "id": "R35"
            },
            {
                "label": "INDEFINITE_TEMPORAL",
                "pattern": [{
                    'LOWER': 'as',
                    'TAG': 'IN'
                }],
                "id": "R35"
            },
            {
                "label": "INDEFINITE_TEMPORAL",
                "pattern": [{
                    'LOWER': 'once',
                    'TAG': 'IN'
                }],
                "id": "R35"
            },
            {
                "label": "INDEFINITE_TEMPORAL",
                "pattern": [{
                    'LOWER': 'earliest'
                }],
                "id": "R35"
            },
            {
                "label": "INDEFINITE_TEMPORAL",
                "pattern": [{
                    'LOWER': 'latest'
                }],
                "id": "R35"
            },
            {
                "label": "INDEFINITE_TEMPORAL",
                "pattern": [{
                    'LOWER': 'instantaneous'
                }],
                "id": "R35"
            },
            {
                "label": "INDEFINITE_TEMPORAL",
                "pattern": [{
                    'LOWER': 'simultaneous'
                }],
                "id": "R35"
            },
            {
                "label": "INDEFINITE_TEMPORAL",
                "pattern": [{
                    'LOWER': 'while'
                }],
                "id": "R35"
            },
            {
                "label": "INDEFINITE_TEMPORAL",
                "pattern": [{
                    'LOWER': 'at'
                }, {
                    'LOWER': 'last'
                }],
                "id": "R35"
            }
        ]
        ruler.add_patterns(patterns)
        nlp.add_pipe(ruler)

        # Process the texts and append the results
        for doc in nlp.pipe(texts):
            ents_json = []
            for ent in doc.ents:
                ents_json.append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'id': ent.ent_id_,
                    'label': ent.label_
                })

            results.append({
                "text":
                doc.text,
                "R1_is_complete":
                doc._.R1_is_complete,
                "R2_no_passive":
                doc._.R2_no_passive,
                "R5_definite_article":
                doc._.R5_definite_article,
                "R6_has_num_quans":
                doc._.R6_has_num_quans,
                "R7_no_vague_terms":
                doc._.R7_no_vague_terms,
                "R8_no_escapeclause":
                not (doc._.has_escapeclause),
                "R9_no_open-ended_clause":
                not (doc._.has_openclause),
                "R10_no_superfluous":
                doc._.R10_no_superfluous,
                "R16_no_not":
                doc._.R16_no_not,
                "R17_no_oblique":
                doc._.R17_no_oblique,
                "R18_single_sent":
                doc._.R18_single_sent,
                "R19_no_combinator":
                not (doc._.has_combinator),
                "R20_no_purpose":
                doc._.R20_no_purpose,
                "R21_no_parenthese":
                doc._.R21_no_parenthese,
                "R24_no_pronoun":
                doc._.R24_no_pronoun,
                "R26_no_unachievable":
                doc._.R26_no_unachievable,
                "R35_no_indefinite_temporal":
                doc._.R35_no_indefinite_temporal,
                "ents":
                ents_json
                # "ents":[{"start": 120, "end": 134, "label": "EscapeClause"}]
            })
            # results.append({"ents":[{'start': 120, 'end': 134, 'label': 'EscapeClause'}]})

    except Exception as e:
        error_message = ["An error occurred: " + str(e)]
        output = {'results': error_message}
        return output, success
    success = True
    output = {'results': results}
    return output, success


# Client POST request received
@app.route('/', methods=['POST', 'GET'])
def main():
    error_message = "error"
    success = False
    user_data = json.loads(globals.request.data)

    out_data = {'results': 'Input data analyzed as below'}

    texts = user_data['text']
    # out_data.update({'text':texts})
    # apply your algorithm and obtain your results
    results, success = validate(texts)
    if (success == True):
        out_data.update(results)

    if success:
        # apply carried out successfully, send a response to the user
        # msg.body = json.dumps({'Results': 'Input data analyzed successfully.'})
        return Response(json.dumps(out_data), mimetype='application/json')
    else:
        return Response({'Error': error_message})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
    # app.run(host='0.0.0.0', port=5001)
