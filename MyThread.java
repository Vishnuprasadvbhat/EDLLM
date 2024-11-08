public class MyThread extends Thread {
//   @Override
  public void run() {
      System.out.println("Thread " + Thread.currentThread() + " is running");
  }

  public static void main(String[] args) {
      for (int i = 0; i < 5; i++) {
          MyThread thread = new MyThread();
          thread.start(); // Starts the thread, calling the run() method
      }
  }
}
