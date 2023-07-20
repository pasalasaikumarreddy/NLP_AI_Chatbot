package com.chatbotnlpai;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.stream.Collectors;

import opennlp.tools.doccat.BagOfWordsFeatureGenerator;
import opennlp.tools.doccat.DoccatFactory;
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.doccat.FeatureGenerator;
import opennlp.tools.lemmatizer.LemmatizerME;
import opennlp.tools.lemmatizer.LemmatizerModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;
import opennlp.tools.util.model.ModelUtil;

public class aichatbot {
	
	private static Map<String, String> questionAnswer = new HashMap<>();
	static {
		questionAnswer.put("greeting", "Hello, how can I help you?");
		questionAnswer.put("bank-inquiry","our bank's name is abc.we provide all the banking services you need.");
		questionAnswer.put("service-inquiry", "we offer wide range of services like different types of accounts,credit and debit cards,all kinds of loans at bearable interest,nation wide atms,etc...");
		questionAnswer.put("conversation-continue", "What else can I help you with?");
		questionAnswer.put("conversation-complete", "Nice chatting with you. Bbye.");
		questionAnswer.put("balance-inquiry", "sms bal enq to 56565 from your registered mobile number.");
		questionAnswer.put("transaction-problems", "contact our customer care @95990*****");
		questionAnswer.put("card-lost", "sms <block> to 56565 from your registered mobile number");
		questionAnswer.put("loan", "our officer will contact you and brief you");
		questionAnswer.put("ATM-problems", "you can avail all the services from our online app:www.abcbank.googleplay.com");
		questionAnswer.put("bank-details", "we are glad to inform you that we provide employment to 5000 people and we have about 1 lakh customers");
		questionAnswer.put("bank-timings", "bank works from 9 AM to 4 PM on working days");
		questionAnswer.put("bank-locations", "we have a wide range of branches in many of the important cities like tirupati,hyderabad,vizag,banglore etc.");
		questionAnswer.put("creating-account", "our officer will contact you and brief you");
		questionAnswer.put("bank-loans ", "our officer will contact you and brief you");
		questionAnswer.put("education-loans ", "our officer will contact you and brief you");
		questionAnswer.put("job-loans", "our officer will contact you and brief you");
	}

	public static void main(String[] args) throws FileNotFoundException, IOException, InterruptedException {
        new aichatbot();
        
		DoccatModel model = trainCategorizerModel();

		Scanner scanner = new Scanner(System.in);
		while (true) {

			System.out.println("customer:");
			String userInput = scanner.nextLine();

			String[] sentences = breakSentences(userInput);

			String answer = "";
			boolean conversationComplete = false;

			for (String sentence : sentences) {

				String[] tokens = tokenizeSentence(sentence);

				String[] posTags = detectPOSTags(tokens);

				String[] lemmas = lemmatizeTokens(tokens, posTags);

				String category = detectCategory(model, lemmas);

				answer = answer + " " + questionAnswer.get(category);

				if ("conversation-complete".equals(category)) {
					conversationComplete = true;
				}
			}

			System.out.println("ABC Bank:" + answer);
			if (conversationComplete) {
				break;
			}

		}
    scanner.close();
	}

	private static DoccatModel trainCategorizerModel() throws FileNotFoundException, IOException {

		InputStreamFactory inputStreamFactory = new MarkableFileInputStreamFactory(new File("faq-categorizer.txt"));
		ObjectStream<String> lineStream = new PlainTextByLineStream(inputStreamFactory, StandardCharsets.UTF_8);
		ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);

		DoccatFactory factory = new DoccatFactory(new FeatureGenerator[] { new BagOfWordsFeatureGenerator() });

		TrainingParameters params = ModelUtil.createDefaultTrainingParameters();
		params.put(TrainingParameters.CUTOFF_PARAM, 0);

		DoccatModel model = DocumentCategorizerME.train("en", sampleStream, params, factory);
		return model;
	}

	private static String detectCategory(DoccatModel model, String[] finalTokens) throws IOException {

		DocumentCategorizerME myCategorizer = new DocumentCategorizerME(model);

		double[] probabilitiesOfOutcomes = myCategorizer.categorize(finalTokens);
		String category = myCategorizer.getBestCategory(probabilitiesOfOutcomes);
		//System.out.println("Category: " + category);

		return category;

	}

	private static String[] breakSentences(String data) throws FileNotFoundException, IOException {

		try (InputStream modelIn = new FileInputStream("en-sent.bin")) {

			SentenceDetectorME myCategorizer = new SentenceDetectorME(new SentenceModel(modelIn));

			String[] sentences = myCategorizer.sentDetect(data);
			//System.out.println("Sentence Detection: " + Arrays.stream(sentences).collect(Collectors.joining(" | ")));

			return sentences;
		}
	}

	private static String[] tokenizeSentence(String sentence) throws FileNotFoundException, IOException {

		try (InputStream modelIn = new FileInputStream("en-token.bin")) {

			TokenizerME myCategorizer = new TokenizerME(new TokenizerModel(modelIn));

			String[] tokens = myCategorizer.tokenize(sentence);
			//System.out.println("Tokenizer : " + Arrays.stream(tokens).collect(Collectors.joining(" | ")));

			return tokens;

		}
	}

	private static String[] detectPOSTags(String[] tokens) throws IOException {

		try (InputStream modelIn = new FileInputStream("en-pos-maxent.bin")) {

			POSTaggerME myCategorizer = new POSTaggerME(new POSModel(modelIn));

			String[] posTokens = myCategorizer.tag(tokens);
			//System.out.println("POS Tags : " + Arrays.stream(posTokens).collect(Collectors.joining(" | ")));

			return posTokens;

		}

	}

	private static String[] lemmatizeTokens(String[] tokens, String[] posTags)
			throws InvalidFormatException, IOException {

		try (InputStream modelIn = new FileInputStream("en-lemmatizer.bin")) {

			LemmatizerME myCategorizer = new LemmatizerME(new LemmatizerModel(modelIn));
			String[] lemmaTokens = myCategorizer.lemmatize(tokens, posTags);
			//System.out.println("Lemmatizer : " + Arrays.stream(lemmaTokens).collect(Collectors.joining(" | ")));

			return lemmaTokens;

		}
	}

}
